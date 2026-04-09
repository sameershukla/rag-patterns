# LangChain Basics: From First Call to Reusable Pipelines

This module introduces the foundational concepts of LangChain through
three progressively built examples.

------------------------------------------------------------------------

## 📂 Structure

langchain-basics/ └── basic/ ├── 01_basic_prompt_model.py ├──
02_prompt_model_parser.py ├── 03_reusable_chain.py

------------------------------------------------------------------------

## 🔹 Example 1: Basic Prompt + Model

### Problem

How to send structured prompts to an LLM.

### Flow

Input → PromptTemplate → Model → AIMessage

### Key Points

-   Uses ChatPromptTemplate
-   Returns AIMessage
-   Access output via: response.content

### Code Concept

chain = prompt \| model

------------------------------------------------------------------------

## 🔹 Example 2: Prompt + Model + Output Parser

### Problem

Model returns object, we want clean string.

### Flow

Input → Prompt → Model → Parser → String

### Key Points

-   Uses StrOutputParser
-   No need for .content
-   Cleaner outputs

### Code Concept

chain = prompt \| model \| parser

------------------------------------------------------------------------

## 🔹 Example 3: Reusable Chain

### Problem

Avoid rebuilding pipeline for every request.

### Flow

Dynamic Input → Prompt → Model → Parser → Output

### Key Points

-   Build once, reuse many times
-   Introduces functions
-   Real-world pattern

------------------------------------------------------------------------

## ⚙️ Setup

pip install langchain langchain-anthropic

Set API Key:

macOS/Linux: export ANTHROPIC_API_KEY="your_key"

Windows: setx ANTHROPIC_API_KEY "your_key"

------------------------------------------------------------------------

## ▶️ Run

python 01_basic_prompt_model.py python 02_prompt_model_parser.py python
03_reusable_chain.py

------------------------------------------------------------------------

## 💡 Key Concept

System prompt → behavior\
User prompt → task

------------------------------------------------------------------------

## 🚀 Summary

01 → Basic call\
02 → Clean output\
03 → Reusable pipeline

LangChain = Prompt → Model → Processing → Output
