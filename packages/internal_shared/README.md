# Shared components package

This package contains shared components that are used across multiple projects.

## Folder structure

- `[ai_models](./ai_models/)`: Contains the AI models used in the projects. ``available_models.py`` contains the list of available models with their metadata.
- `[models](./models/)`: Contains models like request and response models, as well as database models. Based on their usage, they are further divided into subfolders.
- `[utils](./utils/)`: Contains utility functions that are used across the projects.

## Installation

To install the package, run the following command:

```bash
pip install -e /workspace/shared
```

This command can be used during development. If this devcontainer is launched newly, the package will be installed automatically.