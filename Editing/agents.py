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
from PIL import Image
import wandb

# Initialize wandb
wandb.init(project="autogen-image-editing", name="flux-controlnet-editing")


class SharedMessage:
    images: list
    messages: list
    descriptions: list
    step_counter: int

    def __init__(self, image):
        self.images = [image]
        self.messages = []
        self.descriptions = []
        self.step_counter = 0


base_model = "black-forest-labs/FLUX.1-dev"
controlnet_model = "InstantX/FLUX.1-dev-controlnet-canny"
controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
pipe = FluxControlNetPipeline.from_pretrained(
    base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
)
pipe.to("cuda")
image = Image.open('../Data/PittAd/train_images/10000.jpg')
shared_messages = SharedMessage(image)

# Log initial image to wandb
wandb.log({
    "step": 0,
    "initial_image": wandb.Image(image, caption="Initial Image")
})


def image_editing(last_message, control_image, group_chat):
    prompt = last_message.get("content", "")
    sender = last_message.get("name", "")

    # Generate image
    image = pipe(
        prompt,
        control_image=control_image,
        control_guidance_start=0.2,
        control_guidance_end=0.8,
        controlnet_conditioning_scale=1.0,
        num_inference_steps=28,
        guidance_scale=3.5,
    ).images[0]

    img_uri = pil_to_data_uri(image)
    group_chat.messages.append({
        "role": "assistant",
        "name": "image_generator",
        "content": f"[IMAGE_DATA_URI]{img_uri}"
    })
    shared_messages.images.append(image)
    shared_messages.step_counter += 1

    # Log to wandb
    wandb.log({
        "step": shared_messages.step_counter,
        "generated_image": wandb.Image(image, caption=f"Step {shared_messages.step_counter}: {prompt[:100]}..."),
        "prompt": prompt,
        "sender": sender,
    })

    return True


config_list_4o = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4o"],
    },
)

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


# Custom speaker selection function to control the flow
def custom_speaker_selection(last_speaker, group_chat):
    messages = group_chat.messages

    if len(messages) <= 1:
        return planner_agent

    if last_speaker is planner_agent:
        return text_refiner_agent
    elif last_speaker is text_refiner_agent:
        # Generate image after text_refiner
        image_editing(messages[-1], shared_messages.images[-1], group_chat)
        return critic_agent
    elif last_speaker is critic_agent:
        return planner_agent
    else:
        return planner_agent


group_chat = GroupChat(
    agents=[planner_agent, critic_agent, text_refiner_agent],
    messages=[],
    max_round=12,
    speaker_selection_method=custom_speaker_selection,
)

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config={"config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]},
)

# Start the conversation with initial image
initial_img_uri = pil_to_data_uri(image)
initial_message = f"Here is the initial image to edit:\n<img {initial_img_uri}>\n\nPlease analyze and suggest improvements."

# Run the group chat
planner_agent.initiate_chat(
    group_chat_manager,
    message=initial_message,
)

# Finish wandb run
wandb.finish()