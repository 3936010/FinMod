import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from enum import Enum
from pydantic import BaseModel
from typing import Tuple


class ModelProvider(str, Enum):
    """Enum for supported LLM providers"""
    OPENAI = "OpenAI"
    # OLLAMA = "Ollama"  # Commented out as requested
    OLLAMA = "Ollama"
    GEMINI = "Gemini"



class LLMModel(BaseModel):
    """Represents an LLM model configuration"""
    display_name: str
    model_name: str
    provider: ModelProvider

    def to_choice_tuple(self) -> Tuple[str, str, str]:
        """Convert to format needed for questionary choices"""
        return (self.display_name, self.model_name, self.provider.value)
    
    def has_json_mode(self) -> bool:
        """Check if the model supports JSON mode"""
        return True  # Both OpenAI and Ollama support JSON mode
    
    def is_ollama(self) -> bool:
        """Check if the model is an Ollama model"""
        return self.provider == ModelProvider.OLLAMA


# Define available models
AVAILABLE_MODELS = [
    # OpenAI Models
    LLMModel(
        display_name="[openai] gpt-5-nano",
        model_name="gpt-5-nano",
        provider=ModelProvider.OPENAI
    ),

    LLMModel(
        display_name="[openai] gpt-5.2",
        model_name="gpt-5.2",
        provider=ModelProvider.OPENAI
    ),

    # Ollama Models
    # LLMModel(
    #     display_name="[ollama] gpt-oss:20b",
    #     model_name="gpt-oss:20b",
    #     provider=ModelProvider.OLLAMA
    # ),
    # LLMModel(
    #     display_name="[ollama] qwen3:8b",
    #     model_name="qwen3:8b",
    #     provider=ModelProvider.OLLAMA
    # ),

    # Gemini Models
    LLMModel(
        display_name="[google] gemini-2.5-flash",
        model_name="gemini-2.5-flash",
        provider=ModelProvider.GEMINI
    ),
]


def get_model_info(model_name: str) -> LLMModel | None:
    """Get model information by model_name"""
    return next((model for model in AVAILABLE_MODELS if model.model_name == model_name), None)

def get_model(model_name: str, model_provider: ModelProvider | str) -> ChatOpenAI | ChatOllama | None:
    # Convert string to enum if needed
    if isinstance(model_provider, str):
        try:
            model_provider = ModelProvider(model_provider)
        except ValueError:
            raise ValueError(f"Invalid model provider: {model_provider}. Must be one of: {[p.value for p in ModelProvider]}")
    
    if model_provider == ModelProvider.OPENAI:
        # Get and validate API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Print error to console
            print(f"API Key Error: Please make sure OPENAI_API_KEY is set in your .env file.")
            raise ValueError("OpenAI API key not found.  Please make sure OPENAI_API_KEY is set in your .env file.")
        return ChatOpenAI(model=model_name, api_key=api_key)
    elif model_provider == ModelProvider.OLLAMA:
        # Ollama runs locally, no API key needed
        return ChatOllama(model=model_name)
    elif model_provider == ModelProvider.GEMINI:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure GOOGLE_API_KEY is set in your .env file.")
            raise ValueError("Google API key not found. Please make sure GOOGLE_API_KEY is set in your .env file.")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")