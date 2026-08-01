"""
Script para fazer push de prompts otimizados ao LangSmith Prompt Hub.

Este script:
1. Lê os prompts otimizados de prompts/bug_to_user_story_v2.yml
2. Valida os prompts
3. Faz push PÚBLICO para o LangSmith Hub
4. Adiciona metadados (tags, descrição, técnicas utilizadas)

SIMPLIFICADO: Código mais limpo e direto ao ponto.
"""

import os
import sys
from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from utils import load_yaml, check_env_vars, print_section_header

load_dotenv()

# Arquivo com o prompt otimizado
PROMPT_FILE = "prompts/bug_to_user_story_v2.yml"

# Chave raiz dentro do YAML e nome do repositório no Hub
PROMPT_KEY = "bug_to_user_story_v2"


def build_readme(prompt_data: dict) -> str:
    """
    Monta o README publicado junto ao prompt no LangSmith Hub.

    Args:
        prompt_data: Dados do prompt lidos do YAML

    Returns:
        Texto em Markdown com metadados do prompt
    """
    techniques = prompt_data.get("techniques_applied", [])
    version = prompt_data.get("version", "v2")
    description = prompt_data.get("description", "")

    lines = [
        f"# {PROMPT_KEY} ({version})",
        "",
        description,
        "",
        "## Técnicas de Prompt Engineering aplicadas",
        "",
    ]

    lines.extend(f"- {technique}" for technique in techniques)

    lines.extend([
        "",
        "## Variáveis de entrada",
        "",
        "- `bug_report`: descrição bruta do bug reportado",
        "",
        "## Uso",
        "",
        "```python",
        "from langchain import hub",
        "",
        f"prompt = hub.pull(\"<username>/{PROMPT_KEY}\")",
        "chain = prompt | llm",
        "chain.invoke({\"bug_report\": \"...\"})",
        "```",
    ])

    return "\n".join(lines)


def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    """
    Faz push do prompt otimizado para o LangSmith Hub (PÚBLICO).

    Args:
        prompt_name: Nome do prompt
        prompt_data: Dados do prompt

    Returns:
        True se sucesso, False caso contrário
    """
    system_prompt = prompt_data.get("system_prompt", "")
    user_prompt = prompt_data.get("user_prompt", "{bug_report}")

    try:
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_prompt),
        ])
    except Exception as e:
        print(f"   ❌ Erro ao montar o ChatPromptTemplate: {e}")
        print("      Dica: chaves literais no texto devem ser escapadas como {{ }}.")
        return False

    if "bug_report" not in chat_prompt.input_variables:
        print("   ❌ O prompt não expõe a variável obrigatória 'bug_report'.")
        return False

    extra_vars = [v for v in chat_prompt.input_variables if v != "bug_report"]
    if extra_vars:
        print(f"   ❌ Variáveis inesperadas no template: {', '.join(extra_vars)}")
        print("      Apenas 'bug_report' é fornecida pelo dataset de avaliação.")
        return False

    techniques = prompt_data.get("techniques_applied", [])
    tags = list(prompt_data.get("tags", []))

    try:
        url = hub.push(
            prompt_name,
            chat_prompt,
            new_repo_is_public=True,
            new_repo_description=prompt_data.get("description", ""),
            readme=build_readme(prompt_data),
            tags=tags,
        )
    except Exception as e:
        print(f"   ❌ Erro ao publicar no LangSmith Hub: {e}")
        return False

    print(f"   ✓ Publicado: {url}")
    print(f"   ✓ Visibilidade: público")
    print(f"   ✓ Tags: {', '.join(tags) if tags else 'nenhuma'}")
    print(f"   ✓ Técnicas: {', '.join(techniques) if techniques else 'nenhuma'}")

    return True


def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    """
    Valida estrutura básica de um prompt (versão simplificada).

    Args:
        prompt_data: Dados do prompt

    Returns:
        (is_valid, errors) - Tupla com status e lista de erros
    """
    errors = []

    for field in ("description", "system_prompt", "version"):
        if not prompt_data.get(field):
            errors.append(f"Campo obrigatório faltando ou vazio: {field}")

    system_prompt = str(prompt_data.get("system_prompt", ""))
    user_prompt = str(prompt_data.get("user_prompt", ""))

    if "TODO" in system_prompt or "TODO" in user_prompt:
        errors.append("O prompt ainda contém marcações [TODO]")

    if "{bug_report}" not in system_prompt and "{bug_report}" not in user_prompt:
        errors.append("O prompt não usa a variável {bug_report}")

    techniques = prompt_data.get("techniques_applied", [])
    if len(techniques) < 2:
        errors.append(f"Mínimo de 2 técnicas requeridas, encontradas: {len(techniques)}")

    return (len(errors) == 0, errors)


def main():
    """Função principal"""
    print_section_header("PUSH DE PROMPTS OTIMIZADOS PARA O LANGSMITH HUB")

    if not check_env_vars(["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB"]):
        return 1

    username = os.getenv("USERNAME_LANGSMITH_HUB")

    prompts_file = load_yaml(PROMPT_FILE)
    if not prompts_file:
        print(f"❌ Não foi possível carregar {PROMPT_FILE}")
        return 1

    prompt_data = prompts_file.get(PROMPT_KEY)
    if not prompt_data:
        print(f"❌ Chave '{PROMPT_KEY}' não encontrada em {PROMPT_FILE}")
        return 1

    print(f"Validando {PROMPT_FILE}...")
    is_valid, errors = validate_prompt(prompt_data)

    if not is_valid:
        print("❌ Prompt inválido:")
        for error in errors:
            print(f"   - {error}")
        return 1

    print("   ✓ Prompt válido\n")

    prompt_name = f"{username}/{PROMPT_KEY}"
    print(f"Publicando: {prompt_name}")

    if not push_prompt_to_langsmith(prompt_name, prompt_data):
        return 1

    print("\n✅ Push concluído com sucesso!")
    print("\nPróximos passos:")
    print("1. Confira o prompt em https://smith.langchain.com/prompts")
    print("2. Execute a avaliação: python src/evaluate.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
