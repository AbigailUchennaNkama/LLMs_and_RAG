import litellm
from dotenv import load_dotenv
import os
from IPython.display import display
import pandas as pd
from datasets import load_dataset
from litellm import completion

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
#os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

class Model:
    """Class representing different language models using litellm."""
    
    def __init__(self, model_type, model_name=None):
        self.model_type = model_type
        
        if model_type == 'anthropic':
            self.model_name = "claude-3-haiku-20240307"
        elif model_type == 'openai':
            self.model_name = "gpt-3.5-turbo"
        elif model_type == 'groq':
            self.model_name = "groq/llama3-8b-8192"
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def generate(self, prompt, max_tokens=2000, temperature=0.0):
        """Generates a response from the model based on the given prompt."""
        try:
            response = completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error with {self.model_type}: {str(e)}"

class Prompt:
    """Class representing a text summarization prompt."""
    
    def __init__(self, task_description, guidelines, context=""):
        self.task_description = task_description
        self.guidelines = guidelines
        self.context = context

    def generate(self, text_to_summarize):
        """Generates the prompt to be passed to the model."""
        prompt = f"""
        {self.task_description}

        Guidelines:
        {self.guidelines}

        Context:
        {self.context}

        Text to summarize:
        {text_to_summarize}

        Summary:
        """
        return prompt

def summarizer(text, prompt, model, max_tokens=2000, temperature=0.0):
    """Generates a summary using the specified model."""
    full_prompt = prompt.generate(text)
    return model.generate(full_prompt, max_tokens=max_tokens, temperature=temperature)

def load_data(source, **kwargs):
    """Load data from various sources."""
    if source == "huggingface":
        dataset = load_dataset(kwargs['dataset_name'], split=kwargs.get('split', 'train'))
        if 'sample_indices' in kwargs:
            return dataset.select(kwargs['sample_indices'])
        return dataset
    elif source == "csv":
        return pd.read_csv(kwargs['file_path'])
    elif source == "json":
        return pd.read_json(kwargs['file_path'])
    else:
        raise ValueError(f"Unsupported data source: {source}")

def process_articles(dataset_samples, prompts, models):
    """Process multiple articles with the summarizers and return results as a DataFrame."""
    results = []
    
    for index, article in enumerate(dataset_samples):
        result = {
            'original_index': sample_indices[index],  # Add this line
            'short_headline': article['short_headline'],
            'short_text': article['short_text'],
            'full_article': article['article'][:500],       
        }
        
        for prompt_name, prompt in prompts.items():
            for model_name, model in models.items():
                key = f"{model_name}_{prompt_name}"
                result[key] = summarizer(article['article'], prompt, model)

        results.append(result)

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    return df_results

def analyze_results(df_results):
    """Analyze and print results for each article from DataFrame."""
    for i, row in df_results.iterrows():
        print(f"\n--- Article {i + 1} Results (Dataset Index: {row['original_index']}) ---")
        print(f"Original Short Headline: {row['short_headline']}")
        print(f"Original Short Text: {row['short_text']}")
        
        for model in ['openai', 'llama-3']:  # Add 'anthropic' if needed
            print(f"\n{model.capitalize()} Short Headline: {row[f'{model}_short_headline']} (Length: {len(row[f'{model}_short_headline'])})")
            print(f"{model.capitalize()} Short Text: {row[f'{model}_short_text']} (Length: {len(row[f'{model}_short_text'])})")
        
        print(f"\nFull Article:")
        print(row['full_article'] + "...")
        print("-" * 100)


def main():
    # Configuration
    global sample_indices 
    sample_indices = [0, 10, 20, 30, 40, 50]
    dataset_name = "bjoernp/tagesschau-2018-2023"
    
    # Define prompts
    prompts = {
        'short_headline': Prompt(
            task_description="Generate a short headline for the following article.",
            guidelines="The short headline should be 23-28 characters long.",
            context="You are a professional news editor creating concise headlines for articles."
        ),
        'short_text': Prompt(
            task_description="Generate a short text summary for the following article.",
            guidelines="The summary should be 225-258 characters long.",
            context="You are a professional news editor creating brief summaries of articles."
        )
    }

    # Create models
    models = {
        'openai': Model('openai'),
        #'anthropic': Model('anthropic'),
        'llama-3': Model('groq')  # Using 'groq' for llama-3
    }

    # Load data
    dataset_samples = load_data("huggingface", 
                                dataset_name=dataset_name, 
                                sample_indices=sample_indices)

    # Process articles and get results as DataFrame
    df_results = process_articles(dataset_samples, prompts, models)

    # Analyze and print results
    analyze_results(df_results)
    print(df_results.to_dict(orient='records'))

if __name__ == "__main__":
    main()