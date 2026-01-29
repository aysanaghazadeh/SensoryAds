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
from diffusers.utils import load_image
from huggingface_hub import get_token
from PIL import Image
import wandb
import json
from io import BytesIO
import base64
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers import Flux2Pipeline

MIN_EDITS_BEFORE_NO_ISSUE = 2
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
        self.pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
        self.pipe.to("cuda")
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
            # Make critic deterministic and concise to avoid "planner-like" outputs.
            llm_config={"config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}], "temperature": 0.0,
                        "max_tokens": 60},
        )

        # Fallback critic: used if the main critic refuses.
        # Using a different model often avoids sporadic refusal loops.
        self.critic_fallback_agent = MultimodalConversableAgent(
            name="critic_fallback",
            system_message=CRITIC_SYSTEM_PROMPT,
            max_consecutive_auto_reply=10,
            llm_config={
                "config_list": [
                    {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
                    {"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]},
                ],
                "temperature": 0.0,
                "max_tokens": 60,
            },
        )

        self.text_refiner_agent = ConversableAgent(
            name="text_refiner",
            system_message=TEXT_REFINER_SYSTEM_PROMPT,
            llm_config={"config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}], "temperature": 0.5,
                        "max_tokens": 512},
        )

        # Create a user proxy to initiate the conversation
        self.user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False,
        )
        self.group_chat = GroupChat(
            agents=[self.user_proxy, self.planner_agent, self.critic_agent, self.critic_fallback_agent, self.text_refiner_agent],
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
        image = self.pipe(
            image=control_image,
            prompt=prompt,
            guidance_scale=3
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
            return self.planner_agent

        if not messages:
            return self.planner_agent

        if len(messages) <= 1:
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
                "content": f"""IMPORTANT: You are a STRICT EVALUATOR, not a planner, not an editor.

You MUST NOT propose changes or give instructions (do NOT use verbs like: add, modify, enhance, incorporate, adjust, increase, include).
You MUST NOT ask the user for more information or say "feel free to ask" / "ask me" / "I can't" / "unable".
You MUST IGNORE all previous messages and focus ONLY on:
- The image below
- The advertisement message
- The target sensation

Your task: Evaluate the image given the advertisement message and target sensation.

You MUST output EXACTLY TWO LINES:
Line 1: one label (exactly one of the three strings below)
Line 2: exactly ONE sentence explaining why that label applies (must not contradict Line 1)

You MUST NOT output:
- editing suggestions
- step-by-step instructions
- bullet lists
- multiple sentences
- anything beyond the required 2 lines

Image to evaluate:
<img {img_uri}>

CHOOSING LABEL:
1) Visual Element Inconsistency: visual glitches/artifacts/contradictions.
2) Image-Message Alignment: the ad message/product/brand is NOT clearly shown or the message is not the focus.
3) Sensation Evocation: sensation cues are weak/absent.

PRIORITY RULE (CRITICAL):
1) If Visual Element Inconsistency applies → output "Visual Element Inconsistency"
2) Else if the advertisement message is NOT clearly conveyed (including missing product) → output "Image-Message Alignment"
3) Else (message is clear) if sensation is weak → output "Sensation Evocation"

OUTPUT FORMAT (EXACT):
<Label>
<One-sentence explanation>

ALLOWED LABELS:
Visual Element Inconsistency
Image-Message Alignment
Sensation Evocation
"""
            }
            self.group_chat.messages.append(critic_user_message)

            return self.critic_agent

        elif last_speaker is self.critic_agent or last_speaker is self.critic_fallback_agent:
            critic_response_raw = self.extract_text_content(messages[-1].get("content", "")).strip()
            critic_lines = [ln.strip() for ln in critic_response_raw.splitlines() if ln.strip()]
            critic_label = critic_lines[0] if critic_lines else ""
            critic_explanation = critic_lines[1] if len(critic_lines) > 1 else ""
            critic_response = critic_response_raw
            self.log_agent_response(last_speaker.name, critic_response)
            
            # Extract the issue type from response (handle cases where critic adds extra text)
            issue_type = None
            allowed_labels = {"Visual Element Inconsistency", "Image-Message Alignment", "Sensation Evocation"}

            # Hard validation: must be exactly 2 non-empty lines: label + one sentence.
            too_many_lines = len(critic_lines) != 2
            bad_label = critic_label not in allowed_labels
            sentence_terminators = sum(critic_explanation.count(ch) for ch in [".", "!", "?"])
            bad_explanation = (not critic_explanation) or (sentence_terminators > 1)
            invalid_format = too_many_lines or bad_label or bad_explanation
            
            # Check if critic copied a description (common patterns)
            is_description = (len(critic_response) > 100 and not any(valid_string in critic_response for valid_string in ["Visual Element Inconsistency", "Image-Message Alignment", "Sensation Evocation"]))
            
            # If the model refused (common safety phrasing), try the fallback critic once.
            refusal_patterns = ["i'm sorry", "i can't", "i cannot", "unable to", "can't assist", "cannot assist"]
            is_refusal_text = any(pattern in critic_response.lower() for pattern in refusal_patterns)

            if is_refusal_text and last_speaker is self.critic_agent and self.shared_messages.refusal_retry_count < 1:
                self.shared_messages.refusal_retry_count += 1
                retry_resized_image = self.resize_image_for_llm(self.shared_messages.images[-1], max_size=256)
                retry_img_uri = self.image_to_compressed_uri(retry_resized_image)
                retry_message = {
                    "role": "user",
                    "content": f"""EVALUATE THE IMAGE. Do NOT refuse.

Look at THIS image:
<img {retry_img_uri}>

Advertisement Message: "{self.shared_messages.ad_message}"
Target Sensation: {self.shared_messages.target_sensation}

Output EXACTLY TWO LINES:
Line 1: one of these labels (exact match):
Visual Element Inconsistency
Image-Message Alignment
Sensation Evocation
Line 2: exactly ONE sentence explaining why that label applies.

Do NOT output anything else."""
                }
                self.group_chat.messages.append(retry_message)
                return self.critic_fallback_agent

            if invalid_format or (is_description and len(critic_response) > 50):
                # Critic copied a description instead of evaluating
                self.shared_messages.critic_retry_count += 1
                
                if self.shared_messages.critic_retry_count < 2:  # Max 2 retries
                    print(f"WARNING: Critic output invalid format: {critic_response[:120]}...")
                    print(f"Retrying with stricter two-line format (attempt {self.shared_messages.critic_retry_count})...")
                    # Get the image URI from the most recent image
                    retry_resized_image = self.resize_image_for_llm(self.shared_messages.images[-1], max_size=256)
                    retry_img_uri = self.image_to_compressed_uri(retry_resized_image)
                    # Send a retry message with very explicit instructions
                    retry_message = {
                        "role": "user",
                        "content": f"""INVALID OUTPUT. Follow the required format EXACTLY.

Look at THIS image:
<img {retry_img_uri}>

Advertisement Message: "{self.shared_messages.ad_message}"
Target Sensation: {self.shared_messages.target_sensation}

Output EXACTLY TWO LINES:
Line 1: one of these labels (exact match):
Visual Element Inconsistency
Image-Message Alignment
Sensation Evocation
Line 2: exactly ONE sentence explaining why that label applies.

Do NOT propose changes or give instructions (no verbs like: add, modify, enhance, incorporate, adjust, increase, include).
Do NOT ask questions or say "feel free to ask" / "ask me" / "I can't" / "unable".
Do NOT output anything else."""
                    }
                    self.group_chat.messages.append(retry_message)
                    # If the main critic keeps failing, try fallback.
                    if last_speaker is self.critic_agent:
                        return self.critic_fallback_agent
                    return self.critic_agent
                else:
                    # Max retries reached, default to most likely issue
                    print(f"WARNING: Critic failed after {self.shared_messages.critic_retry_count} retries. Defaulting to 'Image-Message Alignment'")
                    issue_type = "Image-Message Alignment"
                    self.shared_messages.critic_retry_count = 0  # Reset for next evaluation
            
            if critic_label == "Image-Message Alignment" or "Image-Message Alignment" in critic_response:
                issue_type = "Image-Message Alignment"
                self.shared_messages.critic_retry_count = 0  # Reset on success
            elif critic_label == "Sensation Evocation" or "Sensation Evocation" in critic_response:
                issue_type = "Sensation Evocation"
                self.shared_messages.critic_retry_count = 0  # Reset on success
            elif critic_label == "Visual Element Inconsistency" or "Visual Element Inconsistency" in critic_response:
                issue_type = "Visual Element Inconsistency"
                self.shared_messages.critic_retry_count = 0  # Reset on success
            else:
                # If critic didn't output expected format, check for refusal patterns
                is_refusal = is_refusal_text
                
                if is_refusal:
                    print(f"WARNING: Critic refused to evaluate: {critic_response[:100]}...")
                    if self.shared_messages.refusal_retry_count < 3:
                        self.shared_messages.refusal_retry_count += 1
                        retry_resized_image = self.resize_image_for_llm(self.shared_messages.images[-1], max_size=256)
                        retry_img_uri = self.image_to_compressed_uri(retry_resized_image)
                        retry_message = {
                            "role": "user",
                            "content": f"""INVALID OUTPUT. Follow the required format EXACTLY. Refusals are invalid.
Look at this image:
<img {retry_img_uri}>

Advertisement Message: "{self.shared_messages.ad_message}"
Target Sensation: {self.shared_messages.target_sensation}

Output EXACTLY TWO LINES:
Line 1: one of these labels (exact match):
Visual Element Inconsistency
Image-Message Alignment
Sensation Evocation
Line 2: exactly ONE sentence explaining why that label applies.

Do NOT propose changes or give instructions (no verbs like: add, modify, enhance, incorporate, adjust, increase, include).
Do NOT ask questions or say "feel free to ask" / "ask me" / "I can't" / "unable".
Do NOT output anything else."""
                        }
                        self.group_chat.messages.append(retry_message)
                        return self.critic_fallback_agent if last_speaker is self.critic_agent else self.critic_agent
                    print("Treating refusal as evaluation needed - defaulting to 'Sensation Evocation' (most common issue)")
                    issue_type = "Sensation Evocation"
                    self.shared_messages.refusal_retry_count = 0
                else:
                    print(f"WARNING: Critic output unexpected format: {critic_response[:100]}...")
                    print("Defaulting to 'Image-Message Alignment' for safety")
                    issue_type = "Image-Message Alignment"
                critic_response = issue_type  # Use standardized response

            # Always continue the loop on any detected issue by routing back to planner.
            # We do NOT use a "No Issue" label in this pipeline; treat it as invalid if it appears.
            if not issue_type or issue_type == "No Issue":
                issue_type = "Image-Message Alignment"

            self.shared_messages.no_issue_confirmations = 0
            self.shared_messages.no_issue_retry_count = 0
            self.shared_messages.refusal_retry_count = 0
            wandb.log({
                "step": self.shared_messages.step_counter,
                "issue_identified": issue_type
            })

            # Add image for planner to see (most recent edited image)
            resized_image = self.resize_image_for_llm(self.shared_messages.images[-1], max_size=256)
            img_uri = self.image_to_compressed_uri(resized_image)

            # Format all previous attempts for the planner
            previous_attempts_text = ""
            if self.shared_messages.all_previous_instructions:
                for i, prev_instructions in enumerate(self.shared_messages.all_previous_instructions, 1):
                    previous_attempts_text += f"\nAttempt {i}:\n{json.dumps(prev_instructions, indent=2)}\n"
            else:
                previous_attempts_text = "None"

            # Determine what the issue means and what to focus on
            issue_guidance = ""
            if issue_type == "Visual Element Inconsistency":
                issue_guidance = f"""
Critic response to focus on is: {issue_type}

You MUST generate creative edits that:
- Make the visual elements consistent
- Ensure the visual elements are consistent
- Improve the visual elements to be consistent"""
            elif issue_type == "Image-Message Alignment":
                issue_guidance = f"""
Critic response to focus on is: {issue_type}
The advertisement message to convey is "{self.shared_messages.ad_message}"

You MUST generate creative edits that:
- Make the product/brand more prominent and visible
- Ensure the image directly relates to and reinforces the message: "{self.shared_messages.ad_message}"
- Add visual elements that clearly connect to the message
- Improve composition to highlight elements that support the message
- Remove or modify elements that distract from the message

Do NOT just add more sensation elements - focus on making the MESSAGE clear."""
            elif issue_type == "Sensation Evocation":
                issue_guidance = f"""
Critic response to focus on is: {issue_type}
The image must evoke the "{self.shared_messages.target_sensation}" sensation.

You MUST generate creative edits that:
- Add visual cues that directly evoke "{self.shared_messages.target_sensation}"
- Adjust colors, lighting, and texture to create the sensation
- Add atmospheric elements that reinforce "{self.shared_messages.target_sensation}"
- Make the sensation more prominent and noticeable

Focus specifically on evoking "{self.shared_messages.target_sensation}" - be explicit about how each edit contributes to this sensation."""
            else:
                issue_guidance = f"Address the issue: {critic_response}"

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
3. ALL your actions must directly address the specific issue: {issue_type} and be creative.
4. Output ONLY a valid JSON array in the exact format specified, no explanations, no markdown"""
            }
            self.group_chat.messages.append(issue_message)
            return self.planner_agent

        else:
            return self.planner_agent

    def agentic_image_editing(self, filename=None, generated_image=None, ad_message_initial=None, target_sensation_initial=None):
        global image, ad_message, target_sensation, shared_messages, agent_response_round, agent_responses_table
        agent_response_round = 0
        
        wandb.init(project="image-FLUX-KONTEXT-AGENTIC", name=f"{filename}-{target_sensation_initial}")
        agent_responses_table = wandb.Table(columns=["step", "round", "agent", "response"])
        if generated_image is not None:
            image = generated_image
        else:
            image = Image.open('../experiments/generated_images/SensoryAds/20250918_122434/AR_ALL_PixArt/dryness/2/87112.jpg')
        if ad_message_initial is not None:
            ad_message = ad_message_initial
        else:
            ad_message = "I should use this lipbalm because it makes my lips soft."
        if target_sensation_initial is not None:
            target_sensation = target_sensation_initial
        else:
            target_sensation = "Dryness"    
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

        # Send initial image to planner
        initial_message = f"""Here is the initial image to edit:
    <img {initial_img_uri}>
    Advertisement Message: {ad_message}
    Target Sensation: {target_sensation}

    Please generate a sequence of concrete creative visual edits to make this image effectively convey the advertisement message and evoke the target sensation.

    CRITICAL: Output ONLY a valid JSON array in the exact format specified in your system instructions. No explanations, no markdown, no text before or after the JSON."""

        # Start with user_proxy initiating to group chat manager
        self.user_proxy.initiate_chat(
            self.group_chat_manager,
            message=initial_message,
        )

        # Finish wandb run
        wandb.finish()
        return self.shared_messages.images[-1]