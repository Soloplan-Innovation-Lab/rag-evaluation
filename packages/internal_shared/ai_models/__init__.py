from enum import Enum

class AvailableModels(str, Enum):
    """
    Enum for available models.
    """

    GPT_35_TURBO = 'gpt-35-turbo'
    GPT_4 = 'gpt-4'
    GPT_4_32K = 'gpt-4-32k'
    GPT_4_TURBO = 'gpt-4-turbo'
    GPT_4O = 'gpt-4o'
    EMBEDDING_3_LARGE = 'text-embedding-3-large'
    EMBEDDING_3_SMALL = 'text-embedding-3-small'
    EMBEDDING_2 = 'text-embedding-2'
