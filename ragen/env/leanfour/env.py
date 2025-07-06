import gym
from gym import spaces
import numpy as np
from datasets import load_dataset
import re
import random
from ragen.env.base import BaseLanguageBasedEnv
from ragen.utils import all_seed
from ragen.env.leanfour.config import LeanfourEnvConfig
import requests
class LeanfourEnv(BaseLanguageBasedEnv):
    def __init__(self, config: LeanfourEnvConfig):
        super(LeanfourEnv, self).__init__()
        self.config = config
        self.dataset = load_dataset(path=self.config.dataset_path, cache_dir=self.config.cache_dir)
        self.current_question_idx = None
        self.current_question = None
        self.correct_answer = None
        self.step_num = None
        self.render_cache = None
        self.lean4_api_session = requests.Session()
        self.lean4_api_session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        self.lean4_api_url = self.config.lean4api_url
        self.reward = None
        self.header = ""
        self.hints = ""
        self.tactics = ""

    # TODO:  lean4 cmd shouble be covered in #ls  #ln    
    def _extract_answer(self, response):
        # Search for text between #ls and #ln
        match = re.search(r"#ls\s*(.*?)\s*#ln", response, re.DOTALL)
        # print("111111",response)
        # print("222222",match)
        if match:
            return match.group(1).strip()
        return None
        
    # set up     
    def reset(self,seed=None, mode=None):
        # Load dataset only when needed
        if not hasattr(self, 'dataset') or self.dataset is None:
            self.dataset = load_dataset(path=self.config.dataset_path, cache_dir=self.config.cache_dir)
            
        dataset = self.dataset[self.config.split]
        with all_seed(seed):
            self.current_question_idx = random.randint(0, len(dataset) - 1)
        question_data = dataset[self.current_question_idx]
        
        # Extract all needed data
        self.current_question = question_data['formal_theorem']
        self.tactics = question_data.get('tactics', "") or ""
        self.hints = question_data.get('hints', "") or ""
        self.header = question_data.get('header', "") or ""
        self.step_num = 0
        self.render_cache = self.current_question

        # Clean up dataset to save memory
        del self.dataset
        self.dataset = None
        
        return self.render_cache
        
    def step(self, action):
        is_correct, is_valid, feedback = self._check_answer(action)
        print(feedback)
        reward = 1.0 / (2 ** self.step_num) if is_correct else 0.0
        if is_correct:
            observation = "Correct!"  +str(feedback)  
            done = True
        else:
            observation = "Incorrect. Please think again." + str(feedback)    
            done = False
        self.step_num += 1
        info = {"action_is_valid": is_valid, "success": is_correct}
        self.render_cache = observation
        return self.render_cache, reward, done, info

    # TODO: add a function to modify answers based on error info 

    
    def _check_answer(self, user_answer):
        """Send user's answer to Lean4 API and check if it's correct based on response."""
        user_answer = user_answer.strip()
        print("LLM answer is :" , user_answer)
        # Send to Lean4 API and wait for response
        try:
            # TODO: Replace with actual Lean4 API call
            response = self.lean4_api_session.post(
                self.lean4_api_url,                    # The URL to send the request to
                json={"code": user_answer},             # Data to send as JSON
                timeout=60      # Request timeout
            ).json()

            # check output first
            feedback= response.get("output","")
            print("111111", feedback)
            if feedback != "":
                is_correct = False
            else:
                is_correct = response.get("success")
        except Exception as e:
            # Handle API errors
            is_correct = False
            feedback = "Error: " + str(e)
        is_valid = bool(user_answer)  # Answer is valid if non-empty
        return is_correct, is_valid, feedback
       

    def render(self):
        return self.render_cache


if __name__ == "__main__":
    # Create the environment configuration
    config = LeanfourEnvConfig(
      
    )
    
    # Initialize the environment
    env = LeanfourEnv(config)
    
    # Reset the environment to get the first question
    print("Question:")
    question = env.reset(seed=43)
    print(env.header)
    print(env.hints)
    print("\nCorrect answer (for testing purposes):")
    print(env.correct_answer)
    
    # Interactive loop for testing
    user_answer = env.header + "theorem not_nota_intro {p : Prop} (h : p) : ¬¬p := by\n  int1ro aaahn\n  contradiction"
    print(user_answer)

    
    # Take a step in the environment with the user's answer
    #breakpoint()
    obs, reward, done, info = env.step(user_answer)
    
    render_test = env.render() 
    print(render_test)
    # Print the results
    print("\nFeedback:", obs)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)
    
    # If the episode is done, reset the environment for a new question
    if done:
        print("\n--- New Question ---")
        question = env.reset()
        print(question)
        print("\nCorrect answer (for testing purposes):")
        print(env.correct_answer)