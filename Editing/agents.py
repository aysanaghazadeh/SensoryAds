import os
import autogen
from Editing.system_messges import *
from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.capabilities.vision_capability import VisionCapability
from autogen.agentchat.contrib.img_utils import get_pil_image, pil_to_data_uri
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen.agentchat.contrib.capabilities import generate_images
import torch
from diffusers import FluxKontextPipeline
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import load_image
from huggingface_hub import get_token
from PIL import Image
import wandb
import json
from io import BytesIO
import base64
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers import Flux2Pipeline
from utils.data.physical_sensations import SENSATION_HIERARCHY, SENSATIONS_PARENT_MAP
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
MIN_EDITS_BEFORE_NO_ISSUE = 2


def _flatten_sensations_all(node):
    """
    Return all sensation strings under a hierarchy node, INCLUDING intermediate categories.
    - dict keys (e.g., "Touch", "Temperature") are included
    - list items (leaf sensations) are included
    """
    if node is None:
        return []
    if isinstance(node, list):
        return [x for x in node if isinstance(x, str)]
    if isinstance(node, dict):
        out = []
        for k, v in node.items():
            if isinstance(k, str):
                out.append(k)
            out.extend(_flatten_sensations_all(v))
        return out
    return []


def _dedupe_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _find_hierarchy_node(tree, node_name):
    """Find a node by key name in nested sensation hierarchy."""
    if not isinstance(tree, dict):
        return None
    if node_name in tree:
        return tree[node_name]
    for v in tree.values():
        if isinstance(v, dict):
            found = _find_hierarchy_node(v, node_name)
            if found is not None:
                return found
    return None


def get_sensation_options(seed_sensation=None):
    """
    Choose a candidate list of sensations.
    - If seed_sensation is provided, return all sensations under its parent category (siblings), INCLUDING categories.
    - Otherwise return all sensations in the hierarchy (INCLUDING categories).
    """
    if seed_sensation:
        parent = SENSATIONS_PARENT_MAP.get(seed_sensation)
        if parent and parent != "root":
            parent_node = _find_hierarchy_node(SENSATION_HIERARCHY, parent)
            options = _flatten_sensations_all(parent_node)
            if options:
                return _dedupe_preserve_order(options)
    return _dedupe_preserve_order(_flatten_sensations_all(SENSATION_HIERARCHY))

class SharedMessage:
    images: list
    messages: list  
    step_counter: int
    ad_message: str
    target_sensation: str
    current_instructions: list
    all_previous_instructions: list  # Track all previous instruction sets
    critic_retry_count: int  # Track critic retry attempts to prevent infinite loops
    no_issue_confirmations: int  # Track No Issue confirmations
    no_issue_retry_count: int  # Track retries when No Issue is too early
    refusal_retry_count: int  # Track retries when critic refuses

    def __init__(self, image, ad_message, target_sensation):
        self.images = [image]
        self.messages = []
        self.step_counter = 0
        self.ad_message = ad_message
        self.target_sensation = target_sensation
        self.current_instructions = []
        self.all_previous_instructions = []  # Initialize history
        self.critic_retry_count = 0  # Initialize retry counter
        self.no_issue_confirmations = 0  # Initialize No Issue confirmation counter
        self.no_issue_retry_count = 0  # Initialize No Issue retry counter
        self.refusal_retry_count = 0  # Initialize refusal retry counter

class ImageEditingAgent:
    def __init__(self, args):
        self.args = args
        self.sensation_options = None
        quantization_config = PipelineQuantizationConfig(
                                    quant_backend="bitsandbytes_8bit",
                                    quant_kwargs={"load_in_8bit": True, "bnb_8bit_quant_type": "nf4", "bnb_8bit_compute_dtype": torch.bfloat16},
                                    components_to_quantize=["transformer", "text_encoder_2"],
                                )
        self.pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", 
                                                        torch_dtype=torch.bfloat16, 
                                                        quantization_config=quantization_config)
        self.pipe = self.pipe.to(device=args.device)
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
        print("pipeline loaded")
        
        self.planner_agent = MultimodalConversableAgent(
            name="planner",
            system_message=PLANNER_SYSTEM_PROMPT,
            max_consecutive_auto_reply=10,
            llm_config={"config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}], "temperature": 0.5,
                        "max_tokens": 512},
        )

        self.critic_agent = MultimodalConversableAgent(
            name="critic",
            system_message=CRITIC_SYSTEM_PROMPT,
            max_consecutive_auto_reply=10,
            llm_config={"config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}], "temperature": 0.5,
                        "max_tokens": 75},  # Enough for the three strings but not too much
        )

        self.text_refiner_agent = ConversableAgent(
            name="text_refiner",
            system_message=TEXT_REFINER_SYSTEM_PROMPT,
            llm_config={"config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}], "temperature": 0.5,
                        "max_tokens": 512},
        )
        
        if args.find_sensation:
            if args.find_AR_message:
                self.sensation_finder_agent = MultimodalConversableAgent(
                    name="sensation_finder",
                    system_message=SENSATION_AND_AR_MESSAGE_FINDER_SYSTEM_PROMPT,
                    max_consecutive_auto_reply=10,
                    llm_config={"config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}], "temperature": 0.5,
                                "max_tokens": 256},
                )
            else:
                self.sensation_finder_agent = MultimodalConversableAgent(
                    name="sensation_finder",
                    system_message=SENSATION_FINDER_SYSTEM_PROMPT,
                    max_consecutive_auto_reply=10,
                    llm_config={"config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}], "temperature": 0.5,
                                "max_tokens": 256},
                )

        # Create a user proxy to initiate the conversation
        self.user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False,
        )
        agents=[self.user_proxy, self.sensation_finder_agent, self.planner_agent, self.critic_agent, self.text_refiner_agent] if getattr(self.args, "find_sensation", False) else [self.user_proxy, self.planner_agent, self.critic_agent, self.text_refiner_agent]
        self.group_chat = GroupChat(
            agents=agents,
            messages=[],
            max_round=20,
            speaker_selection_method=self.custom_speaker_selection,
        )

        self.group_chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config={"config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]},
        )



    def _safe_to_text(self, x, max_len=8000):
        if isinstance(x, str):
            s = x
        else:
            try:
                s = json.dumps(x, ensure_ascii=False)
            except Exception:
                s = str(x)
        s = s.strip()
        return s[:max_len]


    def log_agent_response(self, agent_name, content, extra=None):
        """Persist each agent's response into W&B (table + latest text)."""
        global agent_response_round, agent_responses_table
        agent_response_round += 1
        text = self._safe_to_text(content)
        agent_responses_table.add_data(self.shared_messages.step_counter, agent_response_round, agent_name, text)
        payload = {
            "agent_responses": agent_responses_table,
            "agent_last_speaker": agent_name,
            "agent_response_round": agent_response_round,
            f"{agent_name}_last_response": text[:2000],
        }
        if extra:
            payload.update(extra)
        wandb.log(payload)


    def resize_image_for_llm(self, image, max_size=256):
        """Resize image to reduce token usage while maintaining aspect ratio"""
        width, height = image.size
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            resized = image.resize((new_width, new_height), Image.LANCZOS)
        else:
            resized = image

        # Convert to RGB if necessary and compress as JPEG
        if resized.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', resized.size, (255, 255, 255))
            if resized.mode == 'P':
                resized = resized.convert('RGBA')
            rgb_image.paste(resized, mask=resized.split()[-1] if resized.mode in ('RGBA', 'LA') else None)
            resized = rgb_image

        return resized




    def image_to_compressed_uri(self,image):
        """Convert PIL image to compressed base64 data URI"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=70, optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

    def extract_text_content(self,message_content):
        """Extract text from autogen multimodal content or plain strings."""
        if isinstance(message_content, list):
            for item in message_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text", "")
            return ""
        if isinstance(message_content, str):
            return message_content
        return ""


    def image_editing(self, prompt, control_image, group_chat):
        seed = 0
        positive_magic = {
            "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
            "zh": ", 超清，4K，电影级构图."  # for chinese prompt
        }

        negative_prompt = " "  # using an empty string if you do not have specific concept to remove

        # Generate with different aspect ratios
        aspect_ratios = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1140),
            "3:4": (1140, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        }

        width, height = 512, 512
        # image = self.pipe(
        #     image=control_image,
        #     prompt=prompt,
        #     guidance_scale=3
        # ).images[0]
        image = self.pipe(
            image=control_image,
            prompt=prompt + positive_magic["en"],
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=3,
            num_inference_steps=28,
            true_cfg_scale=4.0,
            generator=torch.Generator(device="cuda").manual_seed(seed)
        ).images[0]

        self.shared_messages.images.append(image)
        self.shared_messages.step_counter += 1

        # Log to wandb
        wandb.log({
            "step": self.shared_messages.step_counter,
            "generated_image": wandb.Image(image, caption=f"Step {self.shared_messages.step_counter}: {prompt[:100]}..."),
            "prompt": prompt,
        })

        return image


    # Custom speaker selection function to control the flow
    def custom_speaker_selection(self, last_speaker, group_chat):
        global planner_response
        messages = group_chat.messages

        # Start with planner after user_proxy sends initial message
        if last_speaker is self.user_proxy:
            if getattr(self.args, "find_sensation", False):
                return self.sensation_finder_agent
            return self.planner_agent

        if not messages:
            return self.planner_agent

        if len(messages) <= 1:
            if getattr(self.args, "find_sensation", False):
                return self.sensation_finder_agent
            return self.planner_agent

        if getattr(self.args, "find_sensation", False) and last_speaker is getattr(self, "sensation_finder_agent", None):
            raw = self.extract_text_content(messages[-1].get("content", "")).strip()
            self.log_agent_response("sensation_finder", raw)

            options = self.sensation_options or []
            # Try to match exactly (case-insensitive)
            choice = None
            raw_norm = raw.strip().lower()
            for opt in options:
                if raw_norm == opt.lower():
                    choice = opt
                    break

            # If not an exact match but the model wrote extra text, find first option substring match
            if choice is None:
                for opt in options:
                    if opt.lower() in raw_norm:
                        choice = opt
                        break

            if choice is None:
                # Retry once with stricter instructions
                retry_msg = {
                    "role": "user",
                    "content": "INVALID OUTPUT. Reply with ONLY ONE sensation from the provided list (verbatim)."
                }
                group_chat.messages.append(retry_msg)
                return self.sensation_finder_agent
            raw_norm = raw_norm.split(':')[-1].strip()
            if getattr(self.args, "find_AR_message", False):
                self.shared_messages.ad_message = raw_norm.split(',')[0].strip()
                self.shared_messages.target_sensation = raw_norm.split(',')[-1].strip()
                wandb.log({"selected_sensation": self.shared_messages.target_sensation, "selected_ad_message": self.shared_messages.ad_message})
            else:
                self.shared_messages.target_sensation = choice
                wandb.log({"selected_sensation": self.shared_messages.target_sensation})

            # Now send the real editing request to planner
            resized_initial_image = self.resize_image_for_llm(self.shared_messages.images[-1], max_size=256)
            initial_img_uri = self.image_to_compressed_uri(resized_initial_image)
            planner_start_message = {
                "role": "user",
                "content": f"""Here is the initial image to edit:
<img {initial_img_uri}>
Advertisement Message: {self.shared_messages.ad_message}
Target Sensation: {self.shared_messages.target_sensation}

Please generate a sequence of concrete visual edits to make this image effectively convey the advertisement message and evoke the target sensation.

CRITICAL: Output ONLY a valid JSON array in the exact format specified in your system instructions. No explanations, no markdown, no text before or after the JSON."""
            }
            group_chat.messages.append(planner_start_message)
            return self.planner_agent

        if last_speaker is self.planner_agent:
            planner_response = messages[-1].get("content", "")
            try:
                planner_response = self.extract_text_content(messages[-1].get("content", ""))
                self.log_agent_response("planner", planner_response)
                
                # Extract JSON from response
                if "```json" in planner_response:
                    json_str = planner_response.split("```json")[1].split("```")[0].strip()
                elif "```" in planner_response:
                    json_str = planner_response.split("```")[1].split("```")[0].strip()
                else:
                    # Try to find JSON array in the response
                    json_str = planner_response.strip()
                    # Remove any leading/trailing text
                    start_idx = json_str.find('[')
                    end_idx = json_str.rfind(']')
                    if start_idx != -1 and end_idx != -1:
                        json_str = json_str[start_idx:end_idx+1]
                
                self.shared_messages.current_instructions = json.loads(json_str)
                # Validate instructions format
                if not isinstance(self.shared_messages.current_instructions, list):
                    raise ValueError("Instructions must be a JSON array")
                if len(self.shared_messages.current_instructions) == 0:
                    raise ValueError("Instructions array cannot be empty")
                
                # Validate action types
                for instruction in self.shared_messages.current_instructions:
                    if not isinstance(instruction, dict):
                        raise ValueError("Each instruction must be a dictionary")
                    if "value" not in instruction:
                        raise ValueError("Each instruction must have a 'value' field")
                
                # Add to history of all previous attempts
                self.shared_messages.all_previous_instructions.append(self.shared_messages.current_instructions.copy())
            except Exception as e:
                self.log_agent_response("planner", planner_response, extra={"planner_parse_error": self._safe_to_text(e)})
                print(f"Error parsing planner instructions: {e}")
                print(f"Planner response was: {planner_response[:500]}...")
                self.shared_messages.current_instructions = []
                # Don't proceed to text_refiner if parsing failed
                # Send a retry message to planner with explicit instructions
                retry_planner_message = {
                    "role": "user",
                    "content": f"""ERROR: Your previous response was not valid JSON. You must output ONLY a JSON array.

The previous response was invalid. Please generate image-editing instructions in JSON format.

Advertisement Message: "{self.shared_messages.ad_message}"
Target Sensation: {self.shared_messages.target_sensation}

Output ONLY a JSON array with this format (nothing else):
[
{{
    "type_of_action": "adding",
    "value": "description"
}}
]

DO NOT output conversational text. DO NOT output explanations. ONLY output the JSON array."""
                }
                group_chat.messages.append(retry_planner_message)
                return self.planner_agent
            
            # Only proceed to text_refiner if we have valid instructions
            if self.shared_messages.current_instructions:
                # Send instructions to text_refiner with explicit conversion instruction
                refiner_message = {
                    "role": "user",
                    "content": f"""Convert these image-editing instructions into a single natural language prompt for image editing.

Instructions (JSON format):
{json.dumps(self.shared_messages.current_instructions, indent=2)}

CRITICAL: Convert the above JSON instructions into ONE cohesive natural language description. 
- Combine all actions into a single flowing text description
- Do NOT output JSON
- Do NOT output markdown or code blocks
- Output ONLY plain text describing what the edited image should look like
- Write in present tense"""
                }
                group_chat.messages.append(refiner_message)
                return self.text_refiner_agent
            else:
                return self.planner_agent

        elif last_speaker is self.text_refiner_agent:
            refined_prompt = self.extract_text_content(messages[-1].get("content", "")).strip()
            self.log_agent_response("text_refiner", refined_prompt)
            new_image = self.image_editing(refined_prompt, self.shared_messages.images[-1], group_chat)
            self.shared_messages.current_description = refined_prompt

            # Resize and compress image before sending to critic
            resized_image = self.resize_image_for_llm(new_image, max_size=256)
            img_uri = self.image_to_compressed_uri(resized_image)
            self.shared_messages.critic_retry_count = 0  # Reset retry counter for new evaluation

            critic_user_message = {
                "role": "user",
                "content": f"""IMPORTANT: You are an STRICT EVALUATOR, not a DESCRIBER.

You MUST NOT copy or paraphrase any previous text or prompts.
You MUST NOT describe the image in full sentences.
You MUST IGNORE all previous messages and focus ONLY on:
- The image below
- The advertisement message
- The target sensation

Your task: Evaluate the image given the advertisement message and target sensation. Output only ONE of the evaluation options AS THE ISSUE OF THE IMAGE AND EXPLAIN WHY YOU CHOSE IT IN ONE SENTENCE.
Strictly follow the following format:
<Issue>
<explanation of the you chose the issue>

Image to evaluate:
<img {img_uri}>

EVALUATION:
1. Are the visual elements in the image consistent? If the visual elements, texutal elements, etc are inconsistent → "Visual Element Inconsistency".
2. Does the image clearly convey "{self.shared_messages.ad_message}"?
- Product/brand visible and prominent?
- Message is the focus?
- CRITICAL: If the specific product mentioned above is NOT clearly visible/recognizable → "Image-Message Alignment" (even if sensation is strong)
- If NO → "Image-Message Alignment"

3. Does the image effectively evoke "{self.shared_messages.target_sensation}"?
- Visual cues evoking the sensation are prominent and strong?
- Sensation is strong?
- If NO → "Sensation Evocation"

PRIORITY RULE (CRITICAL):
1) If Visual Element Inconsistency applies → output "Visual Element Inconsistency"
2) Else if the advertisement message is NOT clearly conveyed (including missing product) → output "Image-Message Alignment"
3) Else (message is clear) if sensation is weak → output "Sensation Evocation"

EXPLAIN WHAT IS THE ISSUE OF THE IMAGE IN ONE SENTENCE AND OUTPUT ONLY ONE OF THE FOLLOWING STRINGS WITHOUT ANY ADDITIONAL TEXT (nothing else):
- Visual Element Inconsistency
- Image-Message Alignment
- Sensation Evocation

"""
            }
            self.group_chat.messages.append(critic_user_message)

            return self.critic_agent

        elif last_speaker is self.critic_agent:
            critic_response = self.extract_text_content(messages[-1].get("content", "")).strip()
            self.log_agent_response("critic", critic_response)
            
            # Extract the issue type from response (handle cases where critic adds extra text)
            issue_type = None
            
            # Check if critic copied a description (common patterns)
            is_description = (len(critic_response) > 100 and not any(valid_string in critic_response for valid_string in ["Visual Element Inconsistency", "Image-Message Alignment", "Sensation Evocation", "No Issue"]))
            
            if is_description and len(critic_response) > 50:
                # Critic copied a description instead of evaluating
                self.shared_messages.critic_retry_count += 1
                
                if self.shared_messages.critic_retry_count < 2:  # Max 2 retries
                    print(f"WARNING: Critic output appears to be a description (copied text): {critic_response[:100]}...")
                    print(f"This is likely copied from text_refiner. Retrying with stronger instructions (attempt {self.shared_messages.critic_retry_count})...")
                    # Get the image URI from the most recent image
                    retry_resized_image = self.resize_image_for_llm(self.shared_messages.images[-1], max_size=256)
                    retry_img_uri = self.image_to_compressed_uri(retry_resized_image)
                    # Send a retry message with very explicit instructions
                    retry_message = {
                        "role": "user",
                        "content": f"""STOP. YOU MADE AN ERROR. You copied text from a previous message. That is WRONG.

YOU ARE AN EVALUATOR. Your job is to EVALUATE the image, NOT describe it, NOT copy text.

COMPLETELY IGNORE all previous messages. DO NOT read them. DO NOT reference them. START FRESH.

Strictly follow the following format:
<Issue>
<explanation of the you chose the issue>

Look at THIS image:
<img {retry_img_uri}>

Advertisement Message: "{self.shared_messages.ad_message}"
Target Sensation: {self.shared_messages.target_sensation}

EVALUATE and output EXACTLY ONE of these strings (nothing else, no descriptions):
- Visual Element Inconsistency
- Image-Message Alignment
- Sensation Evocation

REMEMBER: You are evaluating, not describing. Output only one string followed by the explanation in one sentence."""
                    }
                    self.group_chat.messages.append(retry_message)
                    return self.critic_agent
                else:
                    # Max retries reached, default to most likely issue
                    print(f"WARNING: Critic failed after {self.shared_messages.critic_retry_count} retries. Defaulting to 'Image-Message Alignment'")
                    issue_type = "Image-Message Alignment"
                    self.shared_messages.critic_retry_count = 0  # Reset for next evaluation
            
            if "Image-Message Alignment" in critic_response:
                issue_type = "Image-Message Alignment"
                self.shared_messages.critic_retry_count = 0  # Reset on success
            elif "Sensation Evocation" in critic_response:
                issue_type = "Sensation Evocation"
                self.shared_messages.critic_retry_count = 0  # Reset on success
            elif "Visual Element Inconsistency" in critic_response:
                issue_type = "Visual Element Inconsistency"
                self.shared_messages.critic_retry_count = 0  # Reset on success
            elif "No Issue" in critic_response or "no issue" in critic_response.lower():
                issue_type = "No Issue"
                self.shared_messages.critic_retry_count = 0  # Reset on success
            elif "effectively conveys" in critic_response.lower() and "evokes" in critic_response.lower() and "target sensation" in critic_response.lower():
                # Critic is saying the image is good - map to "No Issue"
                issue_type = "No Issue"
                self.shared_messages.critic_retry_count = 0  # Reset on success
                print(f"INFO: Critic indicated success, mapping to 'No Issue'")
            else:
                # If critic didn't output expected format, check for refusal patterns
                refusal_patterns = ["i'm sorry", "i can't", "i cannot", "unable to", "can't assist"]
                is_refusal = any(pattern in critic_response.lower() for pattern in refusal_patterns)
                
                if is_refusal:
                    print(f"WARNING: Critic refused to evaluate: {critic_response[:100]}...")
                    if self.shared_messages.refusal_retry_count < 1:
                        self.shared_messages.refusal_retry_count += 1
                        retry_resized_image = self.resize_image_for_llm(self.shared_messages.images[-1], max_size=256)
                        retry_img_uri = self.image_to_compressed_uri(retry_resized_image)
                        retry_message = {
                            "role": "user",
                            "content": f"""You MUST evaluate the image if the image based on the instructions in the system prompt, and return the LABEL OF most obvious ISSUE. Reply with exactly ONE label AND EXPLAIN WHY YOU CHOSE IT IN ONE SENTENCE. Refusals are invalid. Do not reply in multiple messages.

Look at this image:
<img {retry_img_uri}>

Advertisement Message: "{self.shared_messages.ad_message}"
Target Sensation: {self.shared_messages.target_sensation}

Strictly follow the following format:
<Issue>
<explanation of the you chose the issue>

Output EXACTLY ONE of these strings and explain why you chose it in one sentence(nothing else):
- Visual Element Inconsistency
- Image-Message Alignment
- Sensation Evocation
"""
                        }
                        self.group_chat.messages.append(retry_message)
                        return self.critic_agent
                    print("Treating refusal as evaluation needed - defaulting to 'Sensation Evocation' (most common issue)")
                    issue_type = "Sensation Evocation"
                    self.shared_messages.refusal_retry_count = 0
                else:
                    print(f"WARNING: Critic output unexpected format: {critic_response[:100]}...")
                    print("Defaulting to 'Image-Message Alignment' for safety")
                    issue_type = "Image-Message Alignment"
                critic_response = issue_type  # Use standardized response

            if "No Issue" in issue_type:
                if self.shared_messages.step_counter < MIN_EDITS_BEFORE_NO_ISSUE:
                    if self.shared_messages.no_issue_retry_count < 1:
                        self.shared_messages.no_issue_retry_count += 1
                        retry_resized_image = self.resize_image_for_llm(self.shared_messages.images[-1], max_size=256)
                        retry_img_uri = self.image_to_compressed_uri(retry_resized_image)
                        retry_message = {
                            "role": "user",
                            "content": f"""No Issue is NOT allowed yet because only {self.shared_messages.step_counter} edit(s) were made.
You MUST choose the single MOST IMPORTANT issue from the three options below.

Look at THIS image:
<img {retry_img_uri}>

Advertisement Message: "{self.shared_messages.ad_message}"
Target Sensation: {self.shared_messages.target_sensation}

Strictly follow the following format:
<Issue>
<explanation of the you chose the issue>

Output EXACTLY ONE of these strings followed by the explanation in one sentence (nothing else):
Visual Element Inconsistency
Image-Message Alignment
Sensation Evocation"""
                        }
                        self.group_chat.messages.append(retry_message)
                        return self.critic_agent
                    issue_type = "Sensation Evocation"
                    self.shared_messages.no_issue_retry_count = 0
                else:
                    self.shared_messages.no_issue_retry_count = 0
                if self.shared_messages.no_issue_confirmations < 1:
                    self.shared_messages.no_issue_confirmations += 1
                    # Ask for a strict confirmation before accepting "No Issue"
                    confirm_resized_image = self.resize_image_for_llm(self.shared_messages.images[-1], max_size=256)
                    confirm_img_uri = self.image_to_compressed_uri(confirm_resized_image)
                    confirm_message = {
                        "role": "user",
                        "content": f"""DOUBLE-CHECK REQUIRED. "No Issue" is only valid if BOTH of these are CLEARLY true:
1) The image clearly conveys the advertisement message.
2) The image strongly evokes the target sensation.
AND there are no visual inconsistencies.

Strictly follow the following format:
<Issue>
<explanation of the you chose the issue>

Look at THIS image:
<img {confirm_img_uri}>

Advertisement Message: "{self.shared_messages.ad_message}"
Target Sensation: {self.shared_messages.target_sensation}

Output EXACTLY ONE of these strings (nothing else):
Visual Element Inconsistency
Image-Message Alignment
Sensation Evocation
"""
                    }
                    self.group_chat.messages.append(confirm_message)
                    return self.critic_agent
                self.shared_messages.no_issue_confirmations = 0
                wandb.log({"final_status": "Success - No Issues"})
                return None
            else:
                self.shared_messages.no_issue_confirmations = 0
                self.shared_messages.no_issue_retry_count = 0
                self.shared_messages.refusal_retry_count = 0
                wandb.log({
                    "step": self.shared_messages.step_counter,
                    "issue_identified": issue_type
                })

                # Add image for planner to see (most recent edited image)
                resized_image = self.resize_image_for_llm(self.shared_messages.images[-1], max_size=256)
                img_uri = self.image_to_compressed_uri(resized_image)  # Use AutoGen's utility for proper format

                # Format all previous attempts for the planner
                previous_attempts_text = ""
                if self.shared_messages.all_previous_instructions:
                    for i, prev_instructions in enumerate(self.shared_messages.all_previous_instructions, 1):
                        previous_attempts_text += f"\nAttempt {i}:\n{json.dumps(prev_instructions, indent=2)}\n"
                else:
                    previous_attempts_text = "None"
                
                # Determine what the issue means and what to focus on
                issue_guidance = ""
                if "Visual Element Inconsistency" in issue_type:
                    issue_guidance = f"""
FOCUS ON: Visual Element Inconsistency Issue
The visual elements in the image are inconsistent.

You MUST generate edits that:
- Make the visual elements consistent
- Ensure the visual elements are consistent
- Improve the visual elements to be consistent"""
                elif "Image-Message Alignment" in issue_type:
                    issue_guidance = f"""
FOCUS ON: Image-Message Alignment Issue
The image does not clearly convey the advertisement message: "{self.shared_messages.ad_message}"

You MUST generate edits that:
- Make the product/brand more prominent and visible
- Ensure the image directly relates to and reinforces the message: "{self.shared_messages.ad_message}"
- Add visual elements that clearly connect to the message
- Improve composition to highlight elements that support the message
- Remove or modify elements that distract from the message

Do NOT just add more sensation elements - focus on making the MESSAGE clear."""
                elif "Sensation Evocation" in issue_type:
                    issue_guidance = f"""
FOCUS ON: Sensation Evocation Issue
The image does not effectively evoke the target sensation: "{self.shared_messages.target_sensation}"

You MUST generate edits that:
- Add visual cues that directly evoke "{self.shared_messages.target_sensation}"
- Adjust colors, lighting, and texture to create the sensation
- Add atmospheric elements that reinforce "{self.shared_messages.target_sensation}"
- Make the sensation more prominent and noticeable

Focus specifically on evoking "{self.shared_messages.target_sensation}" - be explicit about how each edit contributes to this sensation."""
                else:
                    issue_guidance = f"Address the issue: {critic_response}"
                
                # Use string format with image URI for MultimodalConversableAgent
                issue_message = {
                    "role": "user",
                    "content": f"""The critic has identified an issue: {issue_type}

{issue_guidance}

Current Image (most recent edited version):
<img {img_uri}>

Advertisement Message: {self.shared_messages.ad_message}
Target Sensation: {self.shared_messages.target_sensation}

All Previous Instructions Tried (in order, most recent last):
{previous_attempts_text}

CRITICAL REQUIREMENTS:
1. Generate COMPLETELY DIFFERENT editing instructions that have NOT been tried before
2. Look at the previous attempts above and avoid repeating any of those approaches
3. ALL your actions must directly address the specific issue: {issue_type}
4. Output ONLY a valid JSON array in the exact format specified, no explanations, no markdown"""
                }
                self.group_chat.messages.append(issue_message)
                return self.planner_agent

        else:
            return self.planner_agent

    def agentic_image_editing(self, filename, generated_image=None, ad_message_initial=None, target_sensation_initial=None):
        global image, ad_message, target_sensation, shared_messages, agent_response_round, agent_responses_table
        agent_response_round = 0
        if getattr(self.args, "find_AR_message", False) and getattr(self.args, "find_sensation", False):
            wandb.init(project="agentic-sensation-AR-finding-image-generation", name=f"{filename}-{target_sensation_initial}")
        elif getattr(self.args, "find_AR_message", False):
            wandb.init(project="agentic-AR-message-finding-image-generation", name=f"{filename}-{target_sensation_initial}")
        elif getattr(self.args, "find_sensation", False):
            wandb.init(project="agentic-sensation-finding-image-generation", name=f"{filename}-{target_sensation_initial}")
        else:  
            wandb.init(project="agentic-image-genetation_FLUXKontext", name=f"{filename}-{target_sensation_initial}")
        agent_responses_table = wandb.Table(columns=["step", "round", "agent", "response"])
        if generated_image is not None:
            image = generated_image
            if getattr(self.args, "find_sensation", False):
                image = Image.new("RGB", (256, 256), (255, 255, 255))
        else:
            image = Image.open('../experiments/generated_images/SensoryAds/20250918_122434/AR_ALL_PixArt/dryness/2/87112.jpg')
        if ad_message_initial is not None:
            ad_message = ad_message_initial
        else:
            ad_message = "I should use this lipbalm because it makes my lips soft."

        if getattr(self.args, "find_sensation", False):
            # We'll pick the best sensation using the sensation_finder agent.
            self.sensation_options = get_sensation_options(None)
            target_sensation = "UNKNOWN"
        else:
            target_sensation = target_sensation_initial if target_sensation_initial is not None else "Dryness"
        
        

        self.shared_messages = SharedMessage(image, ad_message, target_sensation)

        # Log initial image to wandb
        wandb.log({
            "step": 0,
            "initial_image": wandb.Image(image, caption="Initial Image"),
            "ad_message": ad_message,
            "target_sensation": target_sensation
        })
        # Resize and compress initial image
        resized_initial_image = self.resize_image_for_llm(image, max_size=256)
        initial_img_uri = self.image_to_compressed_uri(resized_initial_image)

        if getattr(self.args, "find_sensation", False):
            options_text = "\n".join([f"- {opt}" for opt in (self.sensation_options or [])])
            if getattr(self.args, "find_AR_message", False):
                initial_message = f"""Advertisement Message: {ad_message}

Select the single best Advertisement Message that is most describable of the possible image and has the strongest message. Next, select the single best Target Sensation to evoke in the image to strengthen this advertisement.

You MUST follow the format strictly:
<explanation of why the advertisement message is the best>
<explanation of why the sensation is the best>
Advertisement Message and Sensation: <Advertisement Message>, <Sensation>

Output ONLY ONE advertisement message and one sensation from this list:
{options_text}
"""
            else:
                initial_message = f"""Advertisement Message: {ad_message}

Select the single best Target Sensation to evoke in the image to strengthen this advertisement.

You MUST follow the format strictly:
<explanation of why the sensation is the best>
The best sensation to evoke is: <Sensation>

Output ONLY ONE sensation from this list:
{options_text}
"""
        else:
            # Send initial image to planner
            initial_message = f"""Here is the initial image to edit:
<img {initial_img_uri}>
Advertisement Message: {ad_message}
Target Sensation: {target_sensation}

Please generate a sequence of concrete visual edits to make this image effectively convey the advertisement message and evoke the target sensation.

CRITICAL: Output ONLY a valid JSON array in the exact format specified in your system instructions. No explanations, no markdown, no text before or after the JSON."""

        # Start with user_proxy initiating to group chat manager
        self.user_proxy.initiate_chat(
            self.group_chat_manager,
            message=initial_message,
        )

        # Finish wandb run
        wandb.finish()
        return self.shared_messages.images[-1]