
import yaml
from jinja2 import Template
from langsmith import Client


def prompt_template_config(yaml_path, prompt_key):

    with open(yaml_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Prompt configuration '{yaml_path}' must be a YAML mapping.")

    prompts = config.get("prompts")
    if not isinstance(prompts, dict):
        raise ValueError(
            f"Prompt configuration '{yaml_path}' must contain a 'prompts' mapping."
        )

    if prompt_key not in prompts:
        raise KeyError(f"Prompt '{prompt_key}' is not defined in '{yaml_path}'.")

    template_content = prompts[prompt_key]
    if not isinstance(template_content, str):
        raise ValueError(f"Prompt '{prompt_key}' in '{yaml_path}' must be a string.")

    return Template(template_content)


def prompt_template_registry(prompt_name):

    ls_client = Client()
    template_content = ls_client.pull_prompt(prompt_name).messages[0].prompt.template

    return Template(template_content)