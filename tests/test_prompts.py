"""
Testes automatizados para validação de prompts.
"""
import pytest
import yaml
import re
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
PROMPT_KEY = "bug_to_user_story_v2"


def load_prompts(file_path: str):
    """Carrega prompts do arquivo YAML."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def prompt_data():
    """Retorna os dados do prompt otimizado (v2)."""
    prompts = load_prompts(str(PROMPT_FILE))

    assert prompts is not None, f"{PROMPT_FILE} está vazio ou inválido"
    assert PROMPT_KEY in prompts, f"Chave '{PROMPT_KEY}' não encontrada em {PROMPT_FILE}"

    return prompts[PROMPT_KEY]


@pytest.fixture(scope="module")
def full_text(prompt_data):
    """Texto completo do prompt (system + user), em minúsculas."""
    system_prompt = prompt_data.get("system_prompt", "")
    user_prompt = prompt_data.get("user_prompt", "")
    return f"{system_prompt}\n{user_prompt}".lower()


class TestPrompts:
    def test_prompt_has_system_prompt(self, prompt_data):
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        assert "system_prompt" in prompt_data, "Campo 'system_prompt' não existe no YAML"

        system_prompt = prompt_data["system_prompt"]

        assert isinstance(system_prompt, str), "'system_prompt' deve ser uma string"
        assert system_prompt.strip(), "'system_prompt' está vazio"
        assert len(system_prompt.strip()) > 200, (
            "'system_prompt' é curto demais para um prompt otimizado "
            f"({len(system_prompt.strip())} caracteres)"
        )

    def test_prompt_has_role_definition(self, prompt_data):
        """Verifica se o prompt define uma persona (ex: "Você é um Product Manager")."""
        system_prompt = prompt_data.get("system_prompt", "")

        assert re.search(r"você é\s+(um|uma)\s+\w+", system_prompt, re.IGNORECASE), (
            "O prompt não define uma persona no formato 'Você é um/uma ...'"
        )

        role_keywords = ["product manager", "product owner", "analista de produto"]
        assert any(keyword in system_prompt.lower() for keyword in role_keywords), (
            f"A persona definida não é de produto. Esperado um de: {role_keywords}"
        )

    def test_prompt_mentions_format(self, full_text):
        """Verifica se o prompt exige formato Markdown ou User Story padrão."""
        assert "markdown" in full_text, "O prompt não exige formato Markdown"

        user_story_parts = ["como um", "eu quero", "para que"]
        missing = [part for part in user_story_parts if part not in full_text]
        assert not missing, (
            f"O prompt não define o template padrão de User Story. Faltando: {missing}"
        )

        assert "critérios de aceitação" in full_text, (
            "O prompt não exige a seção 'Critérios de Aceitação'"
        )

        gherkin_parts = ["dado que", "quando", "então"]
        missing_gherkin = [part for part in gherkin_parts if part not in full_text]
        assert not missing_gherkin, (
            f"O prompt não exige critérios Given-When-Then. Faltando: {missing_gherkin}"
        )

    def test_prompt_has_few_shot_examples(self, prompt_data):
        """Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot)."""
        system_prompt = prompt_data.get("system_prompt", "")
        lowered = system_prompt.lower()

        assert "# exemplos" in lowered or "exemplo 1" in lowered, (
            "O prompt não contém uma seção de exemplos (Few-shot)"
        )

        inputs = len(re.findall(r"###\s*relato de bug", lowered))
        outputs = len(re.findall(r"###\s*user story esperada", lowered))

        assert inputs >= 2, f"Few-shot exige ao menos 2 exemplos de entrada, encontrados: {inputs}"
        assert outputs >= 2, f"Few-shot exige ao menos 2 exemplos de saída, encontrados: {outputs}"
        assert inputs == outputs, (
            f"Cada exemplo precisa de entrada e saída ({inputs} entradas x {outputs} saídas)"
        )

        techniques = [t.lower() for t in prompt_data.get("techniques_applied", [])]
        assert any("few-shot" in t or "few shot" in t for t in techniques), (
            "'Few-shot Learning' não está listado em techniques_applied"
        )

    def test_prompt_no_todos(self, prompt_data):
        """Garante que você não esqueceu nenhum `[TODO]` no texto."""
        placeholders = ["[todo]", "todo:", "to-do", "fixme", "preencher aqui", "<placeholder>"]

        for field, value in prompt_data.items():
            text = str(value).lower()
            found = [p for p in placeholders if p in text]
            assert not found, f"Campo '{field}' ainda contém marcações pendentes: {found}"

        # Nenhuma linha pode ser apenas reticências (esqueleto não preenchido)
        for field, value in prompt_data.items():
            stub_lines = [
                line for line in str(value).splitlines()
                if line.strip() in ("...", "…")
            ]
            assert not stub_lines, f"Campo '{field}' contém linhas não preenchidas ('...')"

    def test_minimum_techniques(self, prompt_data):
        """Verifica (através dos metadados do yaml) se pelo menos 2 técnicas foram listadas."""
        assert "techniques_applied" in prompt_data, (
            "Metadado 'techniques_applied' não existe no YAML"
        )

        techniques = prompt_data["techniques_applied"]

        assert isinstance(techniques, list), "'techniques_applied' deve ser uma lista"
        assert len(techniques) >= 2, (
            f"Mínimo de 2 técnicas requeridas, encontradas: {len(techniques)}"
        )
        assert all(isinstance(t, str) and t.strip() for t in techniques), (
            "Todas as técnicas devem ser strings não vazias"
        )
        assert len(set(techniques)) == len(techniques), (
            "Há técnicas duplicadas em 'techniques_applied'"
        )

    def test_prompt_structure_is_valid(self, prompt_data):
        """Valida a estrutura do prompt com o helper oficial de src/utils.py."""
        is_valid, errors = validate_prompt_structure(prompt_data)
        assert is_valid, f"Estrutura inválida: {errors}"

    def test_prompt_uses_bug_report_variable(self, prompt_data):
        """Garante que o template expõe exatamente a variável esperada pelo dataset."""
        from langchain_core.prompts import ChatPromptTemplate

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_data.get("system_prompt", "")),
            ("human", prompt_data.get("user_prompt", "")),
        ])

        assert chat_prompt.input_variables == ["bug_report"], (
            "O template deve receber apenas 'bug_report', encontrado: "
            f"{chat_prompt.input_variables}"
        )

    def test_prompt_handles_edge_cases(self, prompt_data):
        """Verifica se o prompt documenta tratamento de edge cases."""
        lowered = prompt_data.get("system_prompt", "").lower()

        assert "edge case" in lowered, "O prompt não possui seção de edge cases"
        assert "vago" in lowered or "insuficiente" in lowered, (
            "O prompt não trata relatos vagos ou com informação insuficiente"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
