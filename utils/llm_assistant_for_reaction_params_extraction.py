import pandas as pd
import numpy as np
import openai
import time
import os
from bs4 import BeautifulSoup

def get_elements(html_text):

  # Parse the HTML
  soup = BeautifulSoup(html_text, 'html.parser')

  # Classes of interest
  classes = ["solvents", "ligands", "catalysts", "base"]

  # Initialize an empty list to store highlighted words
  highlighted_words = []

  # Extract words for each class and add to the list
  #for class_name in classes:
  for span in soup.find_all('span'):#, class_=class_name):
          highlighted_words.append(span.text)

  # Display the list of highlighted words
  return highlighted_words


def model_1(txt_file_path):
    """Model 1 will read the text of each paper and extract only the paragraph that refers to the polymerization reaction."""
    response_msgs = []
    file_contents= []
    file_names = []
    answers = ''  
    for filename in os.listdir(txt_file_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(txt_file_path, filename)
            with open(file_path, 'r') as file:               
                file_contents = file.read()

                print("Start to analyze paper: ", {filename})
                user_heading = f"This is a research paper related to polymers synthesis: {file_contents}."
                user_ending = """Your task is read the whole text and identify all the paragraphs that describe the synthetic methods. Your output should be strictly only these
                paragraphs without modifying their context and without adding any other wording."""

                attempts = 5
                while attempts > 0:
                    try:
                        response = client.chat.completions.create(
                            model='gpt-4-turbo-preview',
                            temperature = 0,
                            messages=[{
                                "role": "system",
                                "content": """Answer the question as truthfully as possible using the provided context."""
                            },
                                {"role": "user", "content": user_heading + user_ending}]
                        )
                        answer_str = response.choices[0].message.content
                        print('gpt-answer', answer_str)
                        if not answer_str.lower().startswith("n/a"):
                            answers += '\n' + answer_str
                        break
                    except Exception as e:
                        attempts -= 1
                        if attempts <= 0:
                            print(f"Error: Failed to process paper {filename}. Skipping. (model 1)")
                            break
                        print(f"Error: {str(e)}. Retrying in 60 seconds. {attempts} attempts remaining. (model 1)")
                        time.sleep(60)

        response_msgs.append(answers)
        file_names.append(filename)
    df = pd.concat([pd.DataFrame(file_names, columns=['file_name']), pd.DataFrame(response_msgs, columns=['synthesis paragraphs'])], axis=1)
    return df 

def model_2(df):
    """Model 2 will read the synthesis paragraph and extract all the chemical elements."""
    response_msgs = []
    file_contents= []
    answers = ''  # Collect answers from chatGPT
    # Loop through each file in the folder
    for paragraph in df["synthesis paragraphs"]:
      print("Start to analyze paper: ")
      user_heading = f"This is a paragraph related to polymers synthesis.\n\nContext:\n{paragraph}"
      user_ending = """Your task is to identify all the chemical elements used in the polymerization reaction only. Then generate an HTML version of the input text,
      marking up specific entities related to chemical elements. The specific elements that need to be identified are the following: base, solvents, ligands, and catalysts.
      Use HTML <span> tags to highlight these entities. Each <span> should have a class attribute indicating the type of the entity.
      """
      attempts = 3
      while attempts > 0:
          try:
              response = client.chat.completions.create(
                  model='gpt-4-turbo-preview',
                  temperature = 0,
                  messages=[{
                      "role": "system",
                      "content": """You are a highly intelligent and accurate polymers domain expert.
                       Answer the question as truthfully as possible using the provided context. If you cannot identify the entities return "N/A". """
                  },
                      {"role": "user", "content": user_heading + user_ending}]
              )
              answer_str = response.choices[0].message.content
              break
          except Exception as e:
              attempts -= 1
              if attempts <= 0:
                  print(f"Error: Failed to process paper. Skipping. (model 1)")
                  break
              print(f"Error: {str(e)}. Retrying in 60 seconds. {attempts} attempts remaining. (model 1)")
              time.sleep(60)
      print(answer_str)

      response_msgs.append(answer_str)

    return response_msgs


def model_3(elements_list):
    """Model 3 will evaluate the elements statistics."""
    response_msgs = []
    file_contents= []
    answers = ''  
    user_heading = f"This is a list with all the chemical elemenents used in a polymerization reaction.\n\nContext:\n{elements_list}"
    user_ending = """Your task is to count all the instances and return a dictionary with the main general categories (base, solvents, ligands, catalysts) as keys,
    elements that belong to each category as subkeys and the number of instances as values on the subkeys. Some elements that are similar should be considered as the same
    element, e.g., THF and dry THF are the same elemement and should belong to the same category. Another example FeCl3 and iron (III) cloride is the same element and should belong to the same category.
    Also  Pd(OAc)2 and palladium (II) acetate.
    Also remove from the list any toxic solvent such as cloroform, hexane.

    """
    attempts = 3
    while attempts > 0:
        try:
            response = client.chat.completions.create(
                model='gpt-4-turbo-preview',
                temperature = 0,
                 response_format={ "type": "json_object" },
                messages=[{
                    "role": "system",
                    "content": """You are a highly intelligent and accurate polymers domain expert.
                      Answer the question as truthfully as possible using the provided context and save the results in a json file. """
                },
                    {"role": "user", "content": user_heading + user_ending}]
            )
            answer_str = response.choices[0].message.content

            break
        except Exception as e:
            attempts -= 1
            if attempts <= 0:
                print(f"Error: Failed to process paper. Skipping. (model 1)")
                break
            print(f"Error: {str(e)}. Retrying in 60 seconds. {attempts} attempts remaining. (model 1)")
            time.sleep(60)
    print(answer_str)

    response_msgs.append(answer_str)

    return response_msgs


def main():
    from openai import OpenAI
    client = OpenAI(api_key= "API_KEY") 
    txt_file_path = 'ecps_text' # directory with the txt files from the papers
    model_1_df= model_1(txt_file_path)

    # Splitting the text based on '\n\n'
    paragraphs = model_1_df.split('\n\n')
    ecps_synthesis = pd.DataFrame(paragraphs, columns=['synthesis paragraphs'])
    df_elements = model_2(ecps_synthesis)
    
    elements = []
    for i in range(len(df_elements)):
        elements.append(get_elements(df_elements[i]))
    combined_list = []
    for sublist in elements:
        combined_list.extend(sublist)
    dictionary = model_3(combined_list)
    return dictionary


if __name__ == '__main__':
    main()