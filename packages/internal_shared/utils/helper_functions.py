import os


def get_env_variable(var_name):
    """
    Get an environment variable or raise an exception if it is not set.
    """
    value = os.getenv(var_name)
    if value is None:
        raise EnvironmentError(f"Environment variable {var_name} is not set")
    return value
