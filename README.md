# LangChain Virtual Assistant

LangChain Virtual Assistant

## Setup

1. Clone the repository:

```bash
git clone https://github.com/MartinCastroAlvarez/langchain-virtual-assistant.git
cd langchain-virtual-assistant
```

2. Install dependencies using Poetry:

```bash
poetry install
```
3. Set the OpenAI API Key.

```bash
export OPENAI_API_KEY="lorem-ipsum"
```

## Usage

1. Place your PDF files in the [./pdfs](./pdfs) directory. Alternatively, you can generate test PDFs using the following command:

```bash
poetry run python3 generator.py --number 10
```

2. You can then generate a vector store in [vectorstore.json](vectorstore.json) . using the following command:

```bash
poetry run python3 indexer.py
```

3. Finally, run the smart agent and start asking questions:

```bash
poetry run python3 agent.py
```

## Example

```bash
¡Bienvenido! Soy su asistente médico virtual. ¿En qué puedo ayudarle hoy?
Type one of adiós, bye, chau, exit, goodbye, quit, salir to end.
```

```bash
>>> tengo dolor de cabeza
: Agent: Catalina Ríos fue la paciente con la queja de dolor de cabeza. Esta información se encuentra en el Informe de Consulta Médica de Catalina Ríos, con fecha 02/05/2024.
```

```bash
>>> tengo dolor de hombro
Agent: El paciente con dolor de hombro tiene un diagnóstico de Síndrome del intestino irritable (SII) y se le recomienda programar una ecografía para descartar complicaciones. Este diagnóstico y recomendación se encuentran en el informe de consulta médica de Daniel Ortega en el archivo 'Daniel_Ortega_07-06-2024.pdf'.
```

```bash
>>> tengo ansiedad y depresión
Agent: El paciente con ansiedad y depresión puede encontrar información relevante en los archivos 'Sophia_Ortega_18-03-2025.pdf', 'Valentina_Ríos_14-09-2024.pdf', 'William_Ríos_20-01-2025.pdf', entre otros.
```
