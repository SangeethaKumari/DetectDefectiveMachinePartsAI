"""
Editorial Agent

This module implements an editorial content generator that creates high-quality,
original newsletter articles based on validated news analysis.

The editorial agent produces comprehensive articles including:
- Executive summary and compelling headlines
- Well-structured sections with clear subheadings
- Cohesive arguments based on validated data
- Proper citations from verified sources only
- Fact-check notes and metadata
- Pull quotes and social media snippets
- Complete reference sections

The agent ensures all content is original, factually accurate, and properly
sourced from the validated reports and analysis provided by upstream agents.
"""

# =============================================================================
# IMPORTS
# =============================================================================

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from newsletter.sub_agents.editorial_agent.prompt import EDITORIAL_INSTRUCTION

# =============================================================================
# CONFIGURATION
# =============================================================================

OLLAMA_MODEL = "ollama_chat/qwen3:8b"  # Alternative model for editorial generation

# =============================================================================
# EDITORIAL GENERATION AGENT
# =============================================================================

image_preprocessing_agent = LlmAgent(
    name="image_preprocessing_agent",
    model=LiteLlm(model=OLLAMA_MODEL),
    description="Preprocesses the image for the image classification agent.",
    instruction="Resizes and normalizes image for the tensor model.",
    tools=["image_preprocessing_tool"],
    output_key="image_preprocessed",
)