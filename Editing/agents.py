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


# Initialize wandb
wandb.init(project="autogen-image-editing", name="flux-controlnet-editing")


def resize_image_for_llm(image, max_size=256):
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


def image_to_compressed_uri(image):
    """Convert PIL image to compressed base64 data URI"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=70, optimize=True)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

class SharedMessage:
    images: list
    messages: list
    descriptions: list
    step_counter: int
    ad_message: str
    target_sensation: str
    current_instructions: list
    all_previous_instructions: list  # Track all previous instruction sets
    current_description: str
    critic_retry_count: int  # Track critic retry attempts to prevent infinite loops

    def __init__(self, image, ad_message, target_sensation, initial_description):
        self.images = [image]
        self.messages = []
        self.descriptions = [initial_description]
        self.step_counter = 0
        self.ad_message = ad_message
        self.target_sensation = target_sensation
        self.current_instructions = []
        self.all_previous_instructions = []  # Initialize history
        self.current_description = initial_description
        self.critic_retry_count = 0  # Initialize retry counter


# pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
# pipe.to("cuda")
pipe = Flux2Pipeline.from_pretrained(
    "diffusers/FLUX.2-dev-bnb-4bit", text_encoder=None, torch_dtype=torch.bfloat16, device_map='balanced'
)
print("pipeline loaded")

# Define your image editing task parameters
image = Image.open('../Data/PittAd/train_images/0/10000.jpg')
ad_message = "I should drink this beer because it is refreshing"
target_sensation = "Intense Heat"
initial_description = "A beer advertisement image"

shared_messages = SharedMessage(image, ad_message, target_sensation, initial_description)

# Log initial image to wandb
wandb.log({
    "step": 0,
    "initial_image": wandb.Image(image, caption="Initial Image"),
    "ad_message": ad_message,
    "target_sensation": target_sensation
})


def image_editing(prompt, control_image, group_chat):
    image = pipe(
        image=control_image,
        prompt=prompt,
        guidance_scale=4
    ).images[0]

    shared_messages.images.append(image)
    shared_messages.step_counter += 1

    # Log to wandb
    wandb.log({
        "step": shared_messages.step_counter,
        "generated_image": wandb.Image(image, caption=f"Step {shared_messages.step_counter}: {prompt[:100]}..."),
        "prompt": prompt,
    })

    return image


planner_agent = MultimodalConversableAgent(
    name="planner",
    system_message=PLANNER_SYSTEM_PROMPT,
    max_consecutive_auto_reply=10,
    llm_config={"config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}], "temperature": 0.5,
                "max_tokens": 512},
)

critic_agent = MultimodalConversableAgent(
    name="critic",
    system_message=CRITIC_SYSTEM_PROMPT,
    max_consecutive_auto_reply=10,
    llm_config={"config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}], "temperature": 0.5,
                "max_tokens": 75},  # Enough for the three strings but not too much
)

text_refiner_agent = ConversableAgent(
    name="text_refiner",
    system_message=TEXT_REFINER_SYSTEM_PROMPT,
    llm_config={"config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}], "temperature": 0.5,
                "max_tokens": 512},
)

# Create a user proxy to initiate the conversation
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=30,
    code_execution_config=False,
)


# Custom speaker selection function to control the flow
def custom_speaker_selection(last_speaker, group_chat):
    messages = group_chat.messages

    # Start with planner after user_proxy sends initial message
    if last_speaker is user_proxy:
        return planner_agent

    if len(messages) <= 1:
        return planner_agent

    if last_speaker is planner_agent:
        try:
            planner_response = messages[-1].get("content", "")
            # Handle multimodal content (list format)
            if isinstance(planner_response, list):
                # Extract text from multimodal content
                text_content = ""
                for item in planner_response:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content = item.get("text", "")
                        break
                planner_response = text_content
            
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
            
            shared_messages.current_instructions = json.loads(json_str)
            # Validate instructions format
            if not isinstance(shared_messages.current_instructions, list):
                raise ValueError("Instructions must be a JSON array")
            if len(shared_messages.current_instructions) == 0:
                raise ValueError("Instructions array cannot be empty")
            
            # Validate action types
            valid_action_types = {"adding", "removing", "modifying", "changing_style"}
            for instruction in shared_messages.current_instructions:
                if not isinstance(instruction, dict):
                    raise ValueError("Each instruction must be a dictionary")
                action_type = instruction.get("type_of_action", "")
                if action_type not in valid_action_types:
                    raise ValueError(f"Invalid action type '{action_type}'. Must be one of: {valid_action_types}")
                if "value" not in instruction:
                    raise ValueError("Each instruction must have a 'value' field")
            
            # Add to history of all previous attempts
            shared_messages.all_previous_instructions.append(shared_messages.current_instructions.copy())
        except Exception as e:
            print(f"Error parsing planner instructions: {e}")
            print(f"Planner response was: {planner_response[:500]}...")
            shared_messages.current_instructions = []
            # Don't proceed to text_refiner if parsing failed
            # Send a retry message to planner with explicit instructions
            retry_planner_message = {
                "role": "user",
                "content": f"""ERROR: Your previous response was not valid JSON. You must output ONLY a JSON array.

The previous response was invalid. Please generate image-editing instructions in JSON format.

Advertisement Message: "{shared_messages.ad_message}"
Target Sensation: {shared_messages.target_sensation}

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
            return planner_agent
        
        # Only proceed to text_refiner if we have valid instructions
        if shared_messages.current_instructions:
            # Send instructions to text_refiner with explicit conversion instruction
            refiner_message = {
                "role": "user",
                "content": f"""Convert these image-editing instructions into a single natural language prompt for image editing.

Instructions (JSON format):
{json.dumps(shared_messages.current_instructions, indent=2)}

CRITICAL: Convert the above JSON instructions into ONE cohesive natural language description. 
- Combine all actions into a single flowing text description
- Do NOT output JSON
- Do NOT output markdown or code blocks
- Output ONLY plain text describing what the edited image should look like
- Write in present tense"""
            }
            group_chat.messages.append(refiner_message)
            return text_refiner_agent
        else:
            return planner_agent

    elif last_speaker is text_refiner_agent:
        refined_prompt = messages[-1].get("content", "").strip()
        new_image = image_editing(refined_prompt, shared_messages.images[-1], group_chat)
        shared_messages.current_description = refined_prompt

        # Resize and compress image before sending to critic
        resized_image = resize_image_for_llm(new_image, max_size=256)
        img_uri = image_to_compressed_uri(resized_image)
        shared_messages.critic_retry_count = 0  # Reset retry counter for new evaluation

        critic_user_message = {
            "role": "user",
            "content": f"""CRITICAL: IGNORE ALL PREVIOUS MESSAGES IN THIS CONVERSATION. DO NOT READ, COPY, OR REFERENCE ANY PREVIOUS MESSAGES. START FRESH.

Your task: Evaluate this image and output ONE string.

Image to evaluate:
<img {img_uri}>

Advertisement Message: "{shared_messages.ad_message}"
Target Sensation: {shared_messages.target_sensation}

EVALUATION (ignore all previous conversation):
1. Does the image clearly convey "{shared_messages.ad_message}"?
   - Product/brand visible and prominent?
   - Message is the focus?
   - If NO → "Image-Message Alignment"

2. Does the image effectively evoke "{shared_messages.target_sensation}"?
   - Visual cues are prominent and strong?
   - Sensation is noticeable?
   - If NO → "Sensation Evocation"

3. If BOTH YES → "No Issue"

OUTPUT ONLY ONE STRING (nothing else):
Image-Message Alignment
Sensation Evocation
No Issue"""
        }
        group_chat.messages.append(critic_user_message)

        return critic_agent

    elif last_speaker is critic_agent:
        critic_response = messages[-1].get("content", "").strip()

        # Handle multimodal content (list format)
        if isinstance(critic_response, list):
            text_content = ""
            for item in critic_response:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_content = item.get("text", "")
                    break
            critic_response = text_content.strip()
        
        # Extract the issue type from response (handle cases where critic adds extra text)
        issue_type = None
        
        # Check if critic copied a description (common patterns)
        is_description = any(phrase in critic_response.lower() for phrase in [
            "in the image", "the image features", "the image shows", "the image depicts",
            "the image presents", "the scene showcases", "the scene", "the image",
            "bright sun", "radiant rays", "heat waves", "color palette", "beer bottles",
            "sun-drenched", "sunlit", "vibrant", "saturated", "glow", "thermometer",
            "mirage effects", "sweltering", "glisten", "condensation", "melting ice"
        ]) or (len(critic_response) > 100 and not any(valid_string in critic_response for valid_string in ["Image-Message Alignment", "Sensation Evocation", "No Issue"]))
        
        if is_description and len(critic_response) > 50:
            # Critic copied a description instead of evaluating
            shared_messages.critic_retry_count += 1
            
            if shared_messages.critic_retry_count < 2:  # Max 2 retries
                print(f"WARNING: Critic output appears to be a description (copied text): {critic_response[:100]}...")
                print(f"This is likely copied from text_refiner. Retrying with stronger instructions (attempt {shared_messages.critic_retry_count})...")
                # Get the image URI from the most recent image
                retry_resized_image = resize_image_for_llm(shared_messages.images[-1], max_size=256)
                retry_img_uri = pil_to_data_uri(retry_resized_image)
                # Send a retry message with very explicit instructions
                retry_message = {
                    "role": "user",
                    "content": f"""STOP. YOU MADE AN ERROR. You copied text from a previous message. That is WRONG.

YOU ARE AN EVALUATOR. Your job is to EVALUATE the image, NOT describe it, NOT copy text.

COMPLETELY IGNORE all previous messages. DO NOT read them. DO NOT reference them. START FRESH.

Look at THIS image:
<img {retry_img_uri}>

Advertisement Message: "{shared_messages.ad_message}"
Target Sensation: {shared_messages.target_sensation}

EVALUATE and output EXACTLY ONE of these strings (nothing else, no descriptions):
Image-Message Alignment
Sensation Evocation
No Issue

REMEMBER: You are evaluating, not describing. Output only one string."""
                }
                group_chat.messages.append(retry_message)
                return critic_agent
            else:
                # Max retries reached, default to most likely issue
                print(f"WARNING: Critic failed after {shared_messages.critic_retry_count} retries. Defaulting to 'Sensation Evocation'")
                issue_type = "Sensation Evocation"
                shared_messages.critic_retry_count = 0  # Reset for next evaluation
        
        if "Image-Message Alignment" in critic_response:
            issue_type = "Image-Message Alignment"
            shared_messages.critic_retry_count = 0  # Reset on success
        elif "Sensation Evocation" in critic_response:
            issue_type = "Sensation Evocation"
            shared_messages.critic_retry_count = 0  # Reset on success
        elif "No Issue" in critic_response or "no issue" in critic_response.lower():
            issue_type = "No Issue"
            shared_messages.critic_retry_count = 0  # Reset on success
        elif "effectively conveys" in critic_response.lower() and "evokes" in critic_response.lower() and "target sensation" in critic_response.lower():
            # Critic is saying the image is good - map to "No Issue"
            issue_type = "No Issue"
            shared_messages.critic_retry_count = 0  # Reset on success
            print(f"INFO: Critic indicated success, mapping to 'No Issue'")
        else:
            # If critic didn't output expected format, check for refusal patterns
            refusal_patterns = ["i'm sorry", "i can't", "i cannot", "unable to", "can't assist"]
            is_refusal = any(pattern in critic_response.lower() for pattern in refusal_patterns)
            
            if is_refusal:
                print(f"WARNING: Critic refused to evaluate: {critic_response[:100]}...")
                print("Treating refusal as evaluation needed - defaulting to 'Sensation Evocation' (most common issue)")
                issue_type = "Sensation Evocation"
            else:
                print(f"WARNING: Critic output unexpected format: {critic_response[:100]}...")
                print("Defaulting to 'Image-Message Alignment' for safety")
                issue_type = "Image-Message Alignment"
            critic_response = issue_type  # Use standardized response

        if issue_type == "No Issue":
            wandb.log({"final_status": "Success - No Issues"})
            return None
        else:
            wandb.log({
                "step": shared_messages.step_counter,
                "issue_identified": issue_type
            })

            # Add image for planner to see (most recent edited image)
            resized_image = resize_image_for_llm(shared_messages.images[-1], max_size=256)
            img_uri = pil_to_data_uri(resized_image)  # Use AutoGen's utility for proper format

            # Format all previous attempts for the planner
            previous_attempts_text = ""
            if shared_messages.all_previous_instructions:
                for i, prev_instructions in enumerate(shared_messages.all_previous_instructions, 1):
                    previous_attempts_text += f"\nAttempt {i}:\n{json.dumps(prev_instructions, indent=2)}\n"
            else:
                previous_attempts_text = "None"
            
            # Determine what the issue means and what to focus on
            issue_guidance = ""
            if issue_type == "Image-Message Alignment":
                issue_guidance = f"""
FOCUS ON: Image-Message Alignment Issue
The image does not clearly convey the advertisement message: "{shared_messages.ad_message}"

You MUST generate edits that:
- Make the product/brand more prominent and visible
- Ensure the image directly relates to and reinforces the message: "{shared_messages.ad_message}"
- Add visual elements that clearly connect to the message
- Improve composition to highlight elements that support the message
- Remove or modify elements that distract from the message

Do NOT just add more sensation elements - focus on making the MESSAGE clear."""
            elif issue_type == "Sensation Evocation":
                issue_guidance = f"""
FOCUS ON: Sensation Evocation Issue
The image does not effectively evoke the target sensation: "{shared_messages.target_sensation}"

You MUST generate edits that:
- Add visual cues that directly evoke "{shared_messages.target_sensation}"
- Adjust colors, lighting, and texture to create the sensation
- Add atmospheric elements that reinforce "{shared_messages.target_sensation}"
- Make the sensation more prominent and noticeable

Focus specifically on evoking "{shared_messages.target_sensation}" - be explicit about how each edit contributes to this sensation."""
            else:
                issue_guidance = f"Address the issue: {critic_response}"
            
            # Use string format with image URI for MultimodalConversableAgent
            issue_message = {
                "role": "user",
                "content": f"""The critic has identified an issue: {issue_type}

{issue_guidance}

Current Image (most recent edited version):
<img {img_uri}>

Advertisement Message: {shared_messages.ad_message}
Target Sensation: {shared_messages.target_sensation}

All Previous Instructions Tried (in order, most recent last):
{previous_attempts_text}

CRITICAL REQUIREMENTS:
1. Generate COMPLETELY DIFFERENT editing instructions that have NOT been tried before
2. Look at the previous attempts above and avoid repeating any of those approaches
3. ALL your actions must directly address the specific issue: {issue_type}
4. Output ONLY a valid JSON array in the exact format specified, no explanations, no markdown"""
            }
            group_chat.messages.append(issue_message)
            return planner_agent

    else:
        return planner_agent


group_chat = GroupChat(
    agents=[user_proxy, planner_agent, critic_agent, text_refiner_agent],
    messages=[],
    max_round=15,
    speaker_selection_method=custom_speaker_selection,
)

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config={"config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]},
)


def evoke_sensation():
    # Resize and compress initial image
    resized_initial_image = resize_image_for_llm(image, max_size=256)
    initial_img_uri = image_to_compressed_uri(resized_initial_image)

    # Send initial image to planner
    initial_message = f"""Here is the initial image to edit:
<img {initial_img_uri}>
Advertisement Message: {ad_message}
Target Sensation: {target_sensation}

Please generate a sequence of concrete visual edits to make this image effectively convey the advertisement message and evoke the target sensation.

CRITICAL: Output ONLY a valid JSON array in the exact format specified in your system instructions. No explanations, no markdown, no text before or after the JSON."""

    # Start with user_proxy initiating to group chat manager
    user_proxy.initiate_chat(
        group_chat_manager,
        message=initial_message,
    )

    # Finish wandb run
    wandb.finish()