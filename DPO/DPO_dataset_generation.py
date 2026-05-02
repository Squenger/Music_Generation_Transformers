import torch
import random
import sys
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

try:
    from main_CUDA import MusicGeneratorPipeline
except ModuleNotFoundError:
    # Robust import when script is executed directly from DPO/
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from main_CUDA import MusicGeneratorPipeline

def generate_dpo_dataset(
    maestro_tokens_list, 
    generator_model, 
    judge_model, 
    num_samples=5000, 
    prompt_length=32, 
    generation_length=512,
    batch_size=16 # Ajout de l'argument batch_size
):
    """
    Generates the preference pairs (winner/loser) using the generator model and the judge model.
    """
    generator_model.eval()
    judge_model.eval()
    
    dpo_dataset = []
    
    print(f"generating {num_samples} preference pairs")
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size)):
            
            current_batch = min(batch_size, num_samples - i)
            batch_prompts = []
            
            while len(batch_prompts) < current_batch:
                # pick a random sound from the real dataset 
                real_track = random.choice(maestro_tokens_list)
                if len(real_track) < prompt_length:
                    continue
                    
                start_idx = random.randint(0, len(real_track) - prompt_length)
                prompt_tokens = real_track[start_idx : start_idx + prompt_length]
                batch_prompts.append(prompt_tokens)
            
            
            prompt_tensor = torch.tensor(batch_prompts).cuda()
            
            # let the generator model generate two different suites
            # we use temperature > 1.0 to force the model to try different musical paths
            cont_A = generator_model.generate(prompt_tensor, max_new_tokens=generation_length, temperature=1.5)
            cont_B = generator_model.generate(prompt_tensor, max_new_tokens=generation_length, temperature=1.2)
            
            # extract only the generated parts (excluding the prompt)
            cont_A = cont_A[:,prompt_length:].long()
            cont_B = cont_B[:,prompt_length:].long()

            # judge evaluates both full sequences
            scores_A = judge_model(cont_A).logits.squeeze(-1)
            scores_B = judge_model(cont_B).logits.squeeze(-1)
            
            # determine the winner and loser pour chaque élément du batch
            for b in range(current_batch):
                score_A = scores_A[b].item()
                score_B = scores_B[b].item()
                
                score_diff = abs(score_A - score_B)
                
                # Affichage temporaire pour ajuster le filtre
                tqdm.write(f"[Pair {i+b}] Score A: {score_A:.3f} | Score B: {score_B:.3f} | Diff: {score_diff:.3f}")
                
                if score_diff < 0.01: # we only keep pairs where there is a clear difference in quality
                    continue 
                    
                if score_A > score_B:
                    winner = cont_A[b].cpu().tolist()
                    loser = cont_B[b].cpu().tolist()
                else:
                    winner = cont_B[b].cpu().tolist()
                    loser = cont_A[b].cpu().tolist()
                    
                # save the triplet (prompt, winner, loser)
                dpo_dataset.append({
                    "prompt": batch_prompts[b],
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
    generator_model =MusicGeneratorPipeline(path=None, block_size=512, batch_size=16, n_embd=256, n_head=8, n_layers=8, dropout=0.1)
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
        generator_model = generator_model.model.to("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        judge_model = judge_model.to("mps")
        generator_model = generator_model.model.to("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        judge_model = judge_model.to("cpu")
        generator_model = generator_model.model.to("cpu")
        print("Using CPU")
    
    
    # Ajout du paramètre batch_size=16 ici (tu peux descendre num_samples pour tester l'affichage)
    dpo_dataset = generate_dpo_dataset(full_dataset, generator_model, judge_model, num_samples=5000, batch_size=32)
    print(f"generated {len(dpo_dataset)} preference pairs")