import os
import autogen
from Editing.system_messges import *
from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.capabilities.vision_capability import VisionCapability
from autogen.agentchat.contrib.img_utils import get_pil_image, pil_to_data_uri
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen.agentchat.contrib.capabilities import generate_images
import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline
from diffusers import FluxControlNetModel
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image
from huggingface_hub import get_token
from PIL import Image
import wandb
import json
from io import BytesIO
import base64
from diffusers.quantizers import PipelineQuantizationConfig


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
    current_description: str

    def __init__(self, image, ad_message, target_sensation, initial_description):
        self.images = [image]
        self.messages = []
        self.descriptions = [initial_description]
        self.step_counter = 0
        self.ad_message = ad_message
        self.target_sensation = target_sensation
        self.current_instructions = []
        self.current_description = initial_description


# base_model = "black-forest-labs/FLUX.1-dev"
# controlnet_model = "InstantX/FLUX.1-dev-controlnet-canny"
# controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
# pipe = FluxControlNetPipeline.from_pretrained(
#     base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
# )
# pipe.to("cuda")
quantization_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_8bit",
            quant_kwargs={"load_in_8bit": True, "bnb_8bit_quant_type": "nf4", "bnb_8bit_compute_dtype": torch.bfloat16},
        )
pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511",
                                                  torch_dtype=torch.bfloat16,
                                                  quantization_config=quantization_config,
                                                  device_map='balanced')
device = 'cuda'
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
    # Generate image
    # image = pipe(
    #     prompt,
    #     control_image=control_image,
    #     control_guidance_start=0.2,
    #     control_guidance_end=0.8,
    #     controlnet_conditioning_scale=1.0,
    #     num_inference_steps=28,
    #     guidance_scale=3.5,
    # ).images[0]
    inputs = {
        "image": [control_image],
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 28,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }
    output = pipe(**inputs)
    image = output.images[0]

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
                "max_tokens": 512},
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
    max_consecutive_auto_reply=0,
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
            if "```json" in planner_response:
                json_str = planner_response.split("```json")[1].split("```")[0].strip()
            elif "```" in planner_response:
                json_str = planner_response.split("```")[1].split("```")[0].strip()
            else:
                json_str = planner_response.strip()
            shared_messages.current_instructions = json.loads(json_str)
        except Exception as e:
            print(f"Error parsing planner instructions: {e}")
            shared_messages.current_instructions = []
        return text_refiner_agent

    elif last_speaker is text_refiner_agent:
        refined_prompt = messages[-1].get("content", "").strip()
        new_image = image_editing(refined_prompt, shared_messages.images[-1], group_chat)
        shared_messages.current_description = refined_prompt

        # Resize and compress image before sending to critic
        resized_image = resize_image_for_llm(new_image, max_size=256)
        img_uri = image_to_compressed_uri(resized_image)

        critic_user_message = {
            "role": "user",
            "content": f"""Please evaluate this generated image:
<img {img_uri}>

Advertisement Message: {shared_messages.ad_message}
Target Sensation: {shared_messages.target_sensation}
Applied Instructions: {json.dumps(shared_messages.current_instructions, indent=2)}"""
        }
        group_chat.messages.append(critic_user_message)

        return critic_agent

    elif last_speaker is critic_agent:
        critic_response = messages[-1].get("content", "").strip()

        if "No Issue" in critic_response or "no issue" in critic_response.lower():
            wandb.log({"final_status": "Success - No Issues"})
            return None
        else:
            wandb.log({
                "step": shared_messages.step_counter,
                "issue_identified": critic_response
            })
            return planner_agent

    else:
        return planner_agent


group_chat = GroupChat(
    agents=[user_proxy, planner_agent, critic_agent, text_refiner_agent],
    messages=[],
    max_round=30,
    speaker_selection_method=custom_speaker_selection,
)

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config={"config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]},
)


def evoke_sensation():
    # Send initial context to planner (no image needed for planner)
    initial_message = f"""Here is the task:

Initial Image Description: {initial_description}
Advertisement Message: {ad_message}
Target Sensation: {target_sensation}

Please generate a sequence of concrete visual edits to make this image effectively convey the advertisement message and evoke the target sensation."""

    # Start with user_proxy initiating to group chat manager
    user_proxy.initiate_chat(
        group_chat_manager,
        message=initial_message,
    )

    # Finish wandb run
    wandb.finish()