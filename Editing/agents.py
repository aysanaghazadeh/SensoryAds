# sensory_editing_core_style.py

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Literal, Tuple

from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    default_subscription,
    message_handler,
)

FailureMode = Literal[
    "ok",
    "needs_reedit",
    "needs_replan",
    "prompt_issue",
    "identity_drift",
    "grounding_missing_add",
    "grounding_failed_remove",
]

@dataclass
class Message:
    image_path: str
    target_sensation: str
    constraints: List[str] = field(default_factory=list)

    iter_idx: int = 0
    max_iters: int = 6

    # NEW: descriptions of image states
    original_image_desc: Optional[Dict[str, Any]] = None   # only computed once
    current_image_desc: Optional[Dict[str, Any]] = None    # recomputed after edits if you want

    plan: Optional[Dict[str, Any]] = None
    prompt: Optional[str] = None
    edit_config: Optional[Dict[str, Any]] = None
    edited_image_path: Optional[str] = None
    critic: Optional[Dict[str, Any]] = None

    history: List[Dict[str, Any]] = field(default_factory=list)

@default_subscription
class VisualPlanner(RoutedAgent):
    def __init__(self, plan_fn: Callable[[Message], Dict[str, Any]]) -> None:
        super().__init__("A visual elements planner agent.")
        self._plan_fn = plan_fn

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        # Only plan when needed (first step or needs_replan)
        if message.plan is not None and message.critic is not None:
            if message.critic.get("failure_mode") != "needs_replan":
                return

        plan = self._plan_fn(message)
        message.plan = plan

        print(f"{'-'*80}\nPlanner:\nPlan = {plan}")
        await self.publish_message(message, DefaultTopicId())

@default_subscription
class ImageDescriber(RoutedAgent):
    def __init__(self, describe_fn):
        super().__init__("An image describer agent.")
        self._describe_fn = describe_fn

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        # run only once for the original image
        if message.original_image_desc is not None:
            return

        desc = self._describe_fn(message.image_path)
        message.original_image_desc = desc
        message.current_image_desc = desc  # initialize current state to original

        print(f"{'-'*80}\nDescriber:\nOriginalDesc = {desc}")
        await self.publish_message(message, DefaultTopicId())  # type: ignore

@default_subscription
class PromptRefiner(RoutedAgent):
    def __init__(self, refine_fn: Callable[[Message], Tuple[Dict[str, Any], str]]) -> None:
        super().__init__("A prompt refiner agent (describes current state + writes edit prompt).")
        self._refine_fn = refine_fn

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        if message.plan is None:
            return
        if message.original_image_desc is None:
            return  # wait until describer runs

        # Only regenerate prompt if missing or prompt_issue
        if message.prompt is not None and message.critic is not None:
            if message.critic.get("failure_mode") != "prompt_issue":
                return

        # Optionally refresh current_image_desc every iteration if you want:
        # - if iter_idx > 0 and you have an edited image already
        # Here we do it whenever prompt is generated.
        current_desc, prompt = self._refine_fn(message)
        message.current_image_desc = current_desc
        message.prompt = prompt

        print(f"{'-'*80}\nPromptRefiner:\nCurrentDesc = {current_desc}\nPrompt = {prompt}")
        await self.publish_message(message, DefaultTopicId())  # type: ignore

