import torch
import random
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from main import MusicGeneratorPipeline

def generate_dpo_dataset(
    maestro_tokens_list, 
    generator_model, 
    judge_model, 
    num_samples=5000, 
    prompt_length=64, 
    generation_length=256
):
    """
    Generates the preference pairs (winner/loser) using the generator model and the judge model.
    """
    generator_model.eval()
    judge_model.eval()
    
    dpo_dataset = []
    
    print(f"generating {num_samples} preference pairs")
    
    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            
            # pick a random track from the real dataset 
            real_track = random.choice(maestro_tokens_list)
            if len(real_track) < prompt_length:
                continue
                
            start_idx = random.randint(0, len(real_track) - prompt_length)
            prompt_tokens = real_track[start_idx : start_idx + prompt_length]
            prompt_tensor = torch.tensor(prompt_tokens).unsqueeze(0).cuda()
            
            # let the generator model generate two different continuations
            # we use temperature > 1.0 to force the model to try different musical paths
            cont_A = generator_model.generate(prompt_tensor, max_new_tokens=generation_length, temperature=1)
            cont_B = generator_model.generate(prompt_tensor, max_new_tokens=generation_length, temperature=1)
            
            # extract only the generated parts (excluding the prompt)
            cont_A = cont_A[prompt_length:]
            cont_B = cont_B[prompt_length:]
            
            # judge evaluates both full sequences
            score_A = judge_model(cont_A).logits.item()
            score_B = judge_model(cont_B).logits.item()
            
            # determine the winner and loser
            
            score_diff = abs(score_A - score_B)
            if score_diff < 0.5: # we only keep pairs where there is a clear difference in quality
                continue 
                
            if score_A > score_B:
                winner = cont_A.squeeze(0).cpu().tolist()
                loser = cont_B.squeeze(0).cpu().tolist()
            else:
                winner = cont_B.squeeze(0).cpu().tolist()
                loser = cont_A.squeeze(0).cpu().tolist()
                
            # save the triplet (prompt, winner, loser)
            dpo_dataset.append({
                "prompt": prompt_tokens,
                "winner": winner,
                "loser": loser
            })
            
    # save the dataset to disk 
    torch.save(dpo_dataset, "maximus_dpo_preferences.pt")
    print(f"saved {len(dpo_dataset)} valid preference pairs to disk")
    return dpo_dataset

if __name__ == "__main__":
    # load the full dataset from the file maximus_dpo_preferences.pt
    full_dataset = torch.load(
        r"C:\Users\lab-erima\Documents\Aimine\tests\converting-MidiBERT-into-a-Judge\data_generation\maestro_tokenized_complet.pt",
        weights_only=False
    )
    generator_model =MusicGeneratorPipeline(path=r"C:\Users\lab-erima\Documents\Aimine\tests\Music_Generation_Transformers\data", block_size=64, batch_size=16, n_embd=256, n_head=4, n_layers=4, dropout=0.1)
    generator_model.load_model("output/music_model_best.pth")
    
    judge_model_path = r"C:\\Users\\lab-erima\\Documents\\Aimine\\tests\\converting-MidiBERT-into-a-Judge\\midibert_judge_best"
    try:
        judge_model = AutoModelForSequenceClassification.from_pretrained(
            judge_model_path,
            local_files_only=True
        )
    except Exception:
        judge_model = AutoModelForSequenceClassification.from_pretrained(
            judge_model_path,
            local_files_only=True,
            trust_remote_code=True
        )

    if torch.cuda.is_available():
        judge_model = judge_model.to("cuda")
        generator_model = generator_model.to("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        judge_model = judge_model.to("mps")
        generator_model = generator_model.to("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        judge_model = judge_model.to("cpu")
        generator_model = generator_model.to("cpu")
        print("Using CPU")
    
    
    dpo_dataset = generate_dpo_dataset(full_dataset, generator_model, judge_model, num_samples=5000)
    print(f"generated {len(dpo_dataset)} preference pairs")
