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
import json

# Initialize wandb
wandb.init(project="autogen-image-editing", name="flux-controlnet-editing")


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


base_model = "black-forest-labs/FLUX.1-dev"
controlnet_model = "InstantX/FLUX.1-dev-controlnet-canny"
controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
pipe = FluxControlNetPipeline.from_pretrained(
    base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# Define your image editing task parameters
image = Image.open('../Data/PittAd/train_images/0/10000.jpg')
ad_message = "I should drink this beer because it is refreshing"  # Replace with your actual ad message
target_sensation = "Intense Heat"  # Replace with your actual sensation
initial_description = "A beer advertisement."  # Replace with actual initial description

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
    shared_messages.images.append(image)
    shared_messages.step_counter += 1

    # Log to wandb
    wandb.log({
        "step": shared_messages.step_counter,
        "generated_image": wandb.Image(image, caption=f"Step {shared_messages.step_counter}: {prompt[:100]}..."),
        "prompt": prompt,
    })

    return image, img_uri


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
        # Extract instructions from planner's response
        try:
            planner_response = messages[-1].get("content", "")
            # Extract JSON from response (handle potential markdown formatting)
            if "```json" in planner_response:
                json_str = planner_response.split("```json")[1].split("```")[0].strip()
            elif "```" in planner_response:
                json_str = planner_response.split("```")[1].split("```")[0].strip()
            else:
                json_str = planner_response.strip()
            shared_messages.current_instructions = json.loads(json_str)
        except:
            shared_messages.current_instructions = []
        return text_refiner_agent

    elif last_speaker is text_refiner_agent:
        # Extract refined prompt from text_refiner's response
        refined_prompt = messages[-1].get("content", "").strip()

        # Generate image
        new_image, img_uri = image_editing(refined_prompt, shared_messages.images[-1], group_chat)

        # Update current description (text_refiner should output updated description in second pass)
        shared_messages.current_description = refined_prompt

        # Add image to group chat for critic to see
        group_chat.messages.append({
            "role": "assistant",
            "name": "image_generator",
            "content": f"Image generated. Here is the result:\n<img {img_uri}>"
        })

        return critic_agent

    elif last_speaker is critic_agent:
        # Extract critic's evaluation
        critic_response = messages[-1].get("content", "").strip()

        # If no issue, end the conversation
        if "No Issue" in critic_response or "no issue" in critic_response.lower():
            wandb.log({"final_status": "Success - No Issues"})
            return None
        else:
            # Log the issue and continue
            wandb.log({
                "step": shared_messages.step_counter,
                "issue_identified": critic_response
            })
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

def evoke_sensation():
    # Prepare initial message for planner with all context
    initial_img_uri = pil_to_data_uri(image)
    initial_message = f"""Here is the initial image to edit:
    <img {initial_img_uri}>
    
    Initial Image Description: {initial_description}
    
    Advertisement Message: {ad_message}
    Target Sensation: {target_sensation}
    
    Please generate a sequence of concrete visual edits to make this image effectively convey the advertisement message and evoke the target sensation."""

    # Run the group chat
    planner_agent.initiate_chat(
        group_chat_manager,
        message=initial_message,
    )

    # Finish wandb run
    wandb.finish()