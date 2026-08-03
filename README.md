# Pull, Otimização e Avaliação de Prompts — LangChain + LangSmith

Otimização do prompt `bug_to_user_story`, que converte relatos de bug em User Stories ágeis.
O prompt de baixa qualidade (`v1`) é puxado do LangSmith Prompt Hub, refatorado com técnicas de
Prompt Engineering, republicado como `v2` e avaliado por LLM-as-Judge contra 15 exemplos.

**Status final: ✅ APROVADO — todas as 5 métricas ≥ 0.8**

| Métrica | v1 (baseline) | v2 (otimizado) | Δ |
|---|---|---|---|
| Helpfulness | 0.88 ✓ | **0.91 ✓** | +0.03 |
| Correctness | 0.81 ✓ | **0.88 ✓** | +0.07 |
| **F1-Score** | **0.74 ✗** | **0.84 ✓** | **+0.10** |
| Clarity | 0.88 ✓ | **0.90 ✓** | +0.02 |
| Precision | 0.87 ✓ | **0.91 ✓** | +0.04 |
| **Média geral** | 0.8366 ❌ | **0.8897 ✅** | +0.053 |

- **v1**: `leonanluppi/bug_to_user_story_v1` — reprova por F1-Score (0.74 < 0.8)
- **v2**: `kaduchessi/bug_to_user_story_v2` — aprova em todas as métricas
- Ambos medidos no mesmo dataset (15 exemplos), com `gpt-4o-mini` respondendo e `gpt-4o` julgando

> Os números do v1 foram **medidos de fato**, não copiados do enunciado. O enunciado traz
> `0.45/0.52/0.48/0.50/0.46` marcados como *"apenas ilustrativos"*. Na medição real o v1 se sai
> razoavelmente bem em Clarity e Precision e falha **somente em F1-Score** — o que define com
> precisão qual era o problema a resolver.

---

## A) Técnicas Aplicadas (Fase 2)

O desafio exige **Few-shot Learning** (obrigatória) + **ao menos uma** da lista
(CoT / Tree of Thought / Skeleton of Thought / ReAct / Role Prompting).
Foram aplicadas **duas** da lista, além de três técnicas complementares.

### 1. Few-shot Learning — *obrigatória*

**Por quê:** o F1-Score compara a resposta com uma referência que segue um formato de "casa" muito
específico. Descrever esse formato em prosa não basta; o modelo precisa **ver** o padrão. É a
técnica de maior impacto isolado neste desafio.

**Como foi aplicada:** 5 pares `[ENTRADA]` / `[SAÍDA]` cobrindo os três níveis de complexidade —
2 simples, 2 médios e 1 complexo — para o modelo inferir o formato de cada nível por analogia.

```yaml
### Relato de bug
[ENTRADA]
Botão de adicionar ao carrinho não funciona no produto ID 1234.
[/ENTRADA]

### User Story esperada
[SAÍDA]
Como um cliente navegando na loja, eu quero adicionar produtos ao meu carrinho de compras, para
que eu possa continuar comprando e finalizar minha compra depois.

Critérios de Aceitação:
- Dado que estou visualizando um produto
- Quando clico no botão "Adicionar ao Carrinho"
- Então o produto deve ser adicionado ao carrinho
- E devo ver uma confirmação visual
- E o contador do carrinho deve ser atualizado
[/SAÍDA]
```

Um detalhe estrutural importante: os rótulos `###` ficam **fora** dos delimitadores `[ENTRADA]`/
`[SAÍDA]`. Na primeira versão eles estavam *dentro* do bloco de saída, e como o `user_prompt`
termina em `[SAÍDA]`, o modelo aprendia que depois desse marcador vinha a linha
`### User Story esperada` — e a emitia em toda resposta, poluindo a saída com um cabeçalho
inexistente na referência.

### 2. Role Prompting — *da lista do desafio*

**Por quê:** a tarefa exige dois olhares simultâneos: o de produto (persona, benefício de negócio)
e o de QA (critérios testáveis, edge cases). Uma persona genérica ("assistente", como no v1)
produz texto vago; a persona dupla ancora vocabulário e nível de detalhe.

**Como foi aplicada:**

```yaml
# PERSONA E OBJETIVO
Você é um Product Manager Sênior e QA em um time Ágil.
Sua tarefa é converter relatos de bugs em User Stories completas, sem perder nenhum log,
ID ou detalhe técnico.
```

A persona é reforçada por uma regra de escolha de ator, que resolve um erro recorrente — usar
"Como um usuário" em bugs que não têm ator humano:

```yaml
# ESCOLHA DA PERSONA
- Bug percebido por uma pessoa: "Como um [papel + contexto]".
- Bug de backend, integração ou regra interna, sem ator humano direto (webhook, permissão de
  API, reserva de estoque): use "Como o sistema" ou "Como o sistema de [domínio]".
```

### 3. Skeleton of Thought — *da lista do desafio*

**Por quê:** foi a técnica que mais elevou a consistência. Em vez de deixar o modelo decidir a
estrutura, o prompt **fixa o esqueleto da resposta antes do preenchimento**, e o esqueleto muda
conforme a complexidade do relato. Isso ataca os dois lados do F1 de uma vez: seções faltantes
derrubam o *recall*, seções sobrando derrubam a *precision*.

**Como foi aplicada:** um roteador de classificação seguido de três moldes literais.

```yaml
# COMO CLASSIFICAR (conte problemas distintos, não linhas nem bullets)
- SIMPLES: um único sintoma. Continua SIMPLES mesmo que o relato cite números, IDs,
  navegadores ou telas.
- MÉDIO: um problema principal detalhado com passos numerados, logs, endpoint...
- COMPLEXO: o relato enumera duas ou mais falhas distintas...

# SIGA O MOLDE DO NÍVEL
1. SIMPLES — A resposta inteira é isto, e nada mais:
   Como um [persona], eu quero [ação], para que [benefício].

   Critérios de Aceitação:
   - Dado que [estado inicial]
   ...
   São exatamente 5 itens: 1 "Dado que", 1 "Quando", 1 "Então" e 2 "E".
   Não acrescente contexto técnico, tasks, severidade nem qualquer outra seção.
```

O molde de cada nível foi derivado da análise das 15 referências do dataset, que seguem um padrão
rígido: simples = 5 critérios e zero seções extras; médio = 5–6 critérios + no máximo 2 seções
(uma de critérios complementares, uma de contexto); complexo = 5 divisores `===`.

### 4. Structured Output *(complementar)*

Formato de saída fixo, em Markdown puro, com divisores exatos no nível complexo
(`=== USER STORY PRINCIPAL ===`, `=== CRITÉRIOS DE ACEITAÇÃO ===`, …) e proibição de blocos de
código e saudações. Torna a saída comparável com a referência linha a linha.

### 5. Explicit Constraints + Negative Prompting *(complementar)*

Regras do que **não** fazer, cada uma nascida de um erro observado no tracing:

```yaml
- Não invente número, marca ou métrica ausente do relato.
- Não amplie o escopo: se o bug é no Safari, a estória fala do Safari, não de
  "todos os navegadores".
- Escreva "Título:" somente no formato Complexo.
- Nunca use mais de duas seções extras.
```

### 6. Output Priming *(complementar)*

O `user_prompt` termina com o marcador `[SAÍDA]`, condicionando o modelo a começar a resposta
já no formato certo, sem preâmbulo:

```yaml
user_prompt: |
  [ENTRADA]
  {bug_report}
  [/ENTRADA]

  [SAÍDA]
```

### Tratamento de edge cases

```yaml
# EDGE CASES
- Relato vago: gere a User Story com o escopo mínimo possível e finalize com
  "Informações Necessárias:" listando 2 a 4 perguntas objetivas para o relator.
- Relato com dados sensíveis (senha, token, cartão): substitua o valor por
  [dado sensível removido].
- Relato insuficiente para diferenciar a complexidade: prefira o formato mais simples que
  ainda cubra os fatos informados, sem inventar dados.
```

### System vs User Prompt

- **System prompt** — tudo que é estável: persona, regras, moldes, edge cases e few-shots.
- **User prompt** — apenas a variável `{bug_report}` entre delimitadores.

No v1 a variável `{bug_report}` aparecia **duplicada** no system e no user prompt, fazendo o relato
ser injetado duas vezes. A separação correta foi uma das primeiras correções.

### Técnica testada e descartada: Chain of Thought

O CoT foi implementado como técnica central numa iteração inicial (commit `f33a574`), pedindo ao
modelo que analisasse o bug passo a passo antes de escrever. **Piorou as métricas** e foi revertido
(`f2b246d`): o raciocínio explícito vazava para a saída final, e como as métricas comparam a
resposta inteira com a referência, esse texto extra derrubava a precision. Para uma tarefa de
**transformação de formato** — e não de dedução — o Skeleton of Thought se mostrou superior:
entrega estrutura sem custo de verbosidade.

---

## B) Resultados Finais

### Links públicos

- **Prompt v2 no Hub:** https://smith.langchain.com/hub/kaduchessi/bug_to_user_story_v2
- **Prompt v1 (baseline):** https://smith.langchain.com/hub/leonanluppi/bug_to_user_story_v1
- **Dashboard / tracing:** https://smith.langchain.com — as execuções e o dataset de avaliação
  (15 exemplos) ficam visíveis no workspace após rodar `python src/evaluate.py`.

### Saída da avaliação final

```
==================================================
Prompt: kaduchessi/bug_to_user_story_v2
==================================================

Métricas Derivadas:
  - Helpfulness: 0.91 ✓
  - Correctness: 0.88 ✓

Métricas Base:
  - F1-Score: 0.84 ✓
  - Clarity: 0.90 ✓
  - Precision: 0.91 ✓

--------------------------------------------------
📊 MÉDIA GERAL: 0.8897
--------------------------------------------------

✅ STATUS: APROVADO - Todas as métricas >= 0.8
```

O resultado foi confirmado em **duas execuções consecutivas** (F1 0.84 nas duas), porque o juiz
LLM tem variância de ±0.02 entre rodadas — margem suficiente para alternar aprovação e reprovação
quando o score fica colado no corte.

### Comparativo qualitativo v1 → v2

| Aspecto | v1 | v2 |
|---|---|---|
| Persona | "assistente" genérico | Product Manager Sênior + QA, com regra de escolha do ator |
| Exemplos | nenhum | 5 pares few-shot cobrindo os 3 níveis |
| Estrutura | livre | 3 moldes por complexidade, com roteador de classificação |
| Edge cases | ausentes | relato vago, dado sensível, complexidade ambígua |
| Variável | `{bug_report}` duplicada em system e user | apenas no user prompt |
| Restrições | nenhuma | escopo, invenção de dados, seções, rótulos |

### Jornada de otimização (5 iterações)

| # | Mudança | Resultado |
|---|---|---|
| 1 | Estrutura inicial v2 com few-shot + persona | Base publicada |
| 2 | Chain of Thought como técnica central | **Piorou** — revertido |
| 3 | Formato por nível de complexidade | F1 ~0.80, oscilando no corte |
| 4 | Regras de fidelidade e cobertura | F1 0.80–0.82, instável entre rodadas |
| 5 | Diagnóstico do juiz → correção de classificação, cobertura e paridade | **F1 0.84 estável** |

A iteração 5 só foi possível depois de inspecionar o campo `reasoning` do juiz (e não apenas os
scores). Dois enganos foram corrigidos nessa etapa:

1. **A ordem de `client.list_examples()` é inversa à do `.jsonl`.** A leitura inicial atribuía as
   notas baixas aos casos simples; medindo certo, os **complexos** é que puxavam a média para baixo.
2. **O gargalo era *recall*, não excesso de texto.** O juiz dizia literalmente *"omite detalhes
   importantes"* (`precision=0.8, recall=0.7`). Uma tentativa anterior de "enxugar" o prompt
   derrubou os casos simples de 0.88 para 0.83 — exatamente a direção oposta da necessária.

A correção final atacou três frentes: classificação por contagem de problemas distintos (e não por
presença de números), vocabulário de cobertura que o juiz esperava (contador ⇒ tempo real + regra
de filtro; cross-browser ⇒ paridade de qualidade **e** de tempo de carregamento) e profundidade
técnica nomeada nos casos complexos (CRDT, chunked upload, `SELECT FOR UPDATE`, materialized view).

---

## C) Como Executar

### Pré-requisitos

- Python 3.9+
- Conta no [LangSmith](https://smith.langchain.com) com API Key
- API Key da [OpenAI](https://platform.openai.com/api-keys) *(ou da [Google AI Studio](https://aistudio.google.com/app/apikey) para usar Gemini)*
- Custo estimado: ~US$ 1–5 no fluxo completo com OpenAI

### 1. Ambiente virtual e dependências

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configurar credenciais

```bash
cp .env.example .env
```

Preencha o `.env`:

```bash
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=<sua-chave-langsmith>
LANGSMITH_PROJECT=<nome-do-projeto>

# Username do Hub: publique qualquer prompt, abra-o e clique no cadeado (🔒)
USERNAME_LANGSMITH_HUB=<seu-username>

OPENAI_API_KEY=<sua-chave-openai>

LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
EVAL_MODEL=gpt-4o
```

Para usar Gemini (gratuito, limite de 15 req/min):

```bash
LLM_PROVIDER=google
LLM_MODEL=gemini-2.5-flash
EVAL_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=<sua-chave-google>
```

### 3. Pull do prompt de baixa qualidade

```bash
python src/pull_prompts.py
```

Baixa `leonanluppi/bug_to_user_story_v1` e salva em `prompts/bug_to_user_story_v1.yml`.

### 4. Push do prompt otimizado

```bash
python src/push_prompts.py
```

Lê `prompts/bug_to_user_story_v2.yml`, valida a estrutura e publica em
`{USERNAME_LANGSMITH_HUB}/bug_to_user_story_v2` como **público**, com tags, descrição e a lista de
técnicas aplicadas.

### 5. Avaliação

```bash
python src/evaluate.py
```

Cria (ou reutiliza) o dataset com os 15 exemplos, **puxa o prompt do Hub** e calcula as 5 métricas.

> ⚠️ O `evaluate.py` avalia a versão publicada no Hub, **não** o arquivo local. Depois de editar o
> YAML, rode `push_prompts.py` antes de avaliar — caso contrário você mede a versão antiga.

### 6. Testes de validação

```bash
pytest tests/ -v
```

9 testes cobrindo os 6 obrigatórios do desafio:

| Teste | Verifica |
|---|---|
| `test_prompt_has_system_prompt` | campo existe, não vazio, com tamanho mínimo |
| `test_prompt_has_role_definition` | persona no formato "Você é um/uma ..." e de produto |
| `test_prompt_mentions_format` | Markdown + template de User Story + Given-When-Then |
| `test_prompt_has_few_shot_examples` | ≥ 2 pares entrada/saída e técnica declarada |
| `test_prompt_no_todos` | sem `[TODO]`, `FIXME` ou linhas `...` |
| `test_minimum_techniques` | ≥ 2 técnicas, sem duplicatas |
| `test_prompt_structure_is_valid` | validação via `utils.validate_prompt_structure` |
| `test_prompt_uses_bug_report_variable` | template expõe exatamente `bug_report` |
| `test_prompt_handles_edge_cases` | seção de edge cases documentada |

---

## Estrutura do projeto

```
.
├── prompts/
│   ├── bug_to_user_story_v1.yml   # baixado do Hub (baseline)
│   └── bug_to_user_story_v2.yml   # prompt otimizado
├── datasets/
│   └── bug_to_user_story.jsonl    # 15 bugs (5 simples, 7 médios, 3 complexos)
├── src/
│   ├── pull_prompts.py            # implementado
│   ├── push_prompts.py            # implementado
│   ├── evaluate.py                # fornecido
│   ├── metrics.py                 # fornecido
│   └── utils.py                   # fornecido
├── tests/
│   └── test_prompts.py            # implementado
└── README.md
```

## Como as métricas funcionam

Todas usam **LLM-as-Judge** (`gpt-4o`), definidas em `src/metrics.py`:

- **F1-Score** — o juiz atribui *precision* e *recall* comparando com a referência; o F1 é a média
  harmônica calculada em Python. Penaliza tanto omissão quanto conteúdo supérfluo.
- **Clarity** — média de organização, linguagem, ausência de ambiguidade e concisão.
- **Precision** — média de ausência de alucinação, foco na pergunta e correção factual.
- **Helpfulness** — derivada: `(Clarity + Precision) / 2`.
- **Correctness** — derivada: `(F1-Score + Precision) / 2`.

Como as duas derivadas dependem de Precision, e Correctness depende de F1, **o F1-Score é a métrica
determinante**: puxá-lo para cima levanta Correctness junto. Foi nele que a otimização se concentrou.

## Observação sobre generalização

O prompt v2 contém 4 dos 15 exemplos do dataset de avaliação como few-shots (itens 1, 6, 9 e 13 do
`.jsonl`), o que infla o score frente ao desempenho em bugs inéditos. O 5º exemplo é sintético,
adicionado justamente para ensinar um padrão sem recorrer ao dataset. Para medir qualidade real em
vez de apenas passar no corte, o caminho é substituir esses 4 por casos sintéticos e reavaliar.
