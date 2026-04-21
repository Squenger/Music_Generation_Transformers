import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from miditok import REMI, TokSequence
import symusic
from symusic import Score
import pandas as pd
import os
import glob

class MusicDataset:
    def __init__(self, path, block_size, batch_size, split_ratio=0.9):
        self.tokenizer = REMI()
        self.block_size = block_size
        self.batch_size = batch_size

        path = os.path.expanduser(path)
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"MIDI data directory not found: {path}")

        midi_files = glob.glob(os.path.join(path, "**", "*.midi"), recursive=True)
        midi_files += glob.glob(os.path.join(path, "**", "*.mid"), recursive=True)
        if len(midi_files) == 0:
            raise ValueError(f"No MIDI files found in: {path}")

        all_tokens = []
        
        for idx, file in enumerate(midi_files):
            try:
                midi = Score(file)
                tokens = self.tokenizer(midi)
                
                if isinstance(tokens, list):
                    track_ids = tokens[0].ids
                else:
                    track_ids = tokens.ids
                    
                all_tokens.extend(track_ids)
                
            except Exception as e:

                continue
                
            if (idx + 1) % 50 == 0:
                print(f"{idx + 1}/{len(midi_files)} fichiers tokenisés")

        self.vocab_size = len(self.tokenizer)
        print(f"{len(all_tokens)} evenements musicaux")
        
        data = torch.tensor(all_tokens, dtype=torch.long)
        
        # train / validation
        n = int(split_ratio * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        
    def get_batch(self, section, device):
        d = self.train_data if section == "train" else self.val_data
        i = torch.randint(len(d) - self.block_size, (self.batch_size,))
        x = torch.stack([d[j:j+self.block_size] for j in i])
        y = torch.stack([d[j+1:j+self.block_size+1] for j in i])
        return x.to(device), y.to(device)


class SingleHead(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # (batch , time , head_size)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        # ancienne méthode sur mac  
        # attention = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        # attention = attention.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # attention = F.softmax(attention, dim=-1)
        # attention = self.dropout(attention)
        # out = attention @ v
        
        #FlashAttention sur CUDA
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0)
        
        return out

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = nn.ModuleList([SingleHead(n_embd, head_size, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.proj_dropout = nn.Dropout(dropout)
        self.ffw = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        sa_out = torch.cat([h(self.ln1(x)) for h in self.sa], dim=-1)
        x = x + self.proj_dropout(self.proj(sa_out))
        x = x + self.ffw(self.ln2(x))
        return x

class MusicGEN(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layers, device, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.device = device
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.positional = nn.Embedding(block_size, n_embd)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.final = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, target=None):
        B, T = idx.shape
        tok_embd = self.embedding(idx)  # (B,T,n_embd)
        pos_embd = self.positional(torch.arange(T, device=self.device))  # (T,n_embd)
        x = self.emb_dropout(tok_embd + pos_embd)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.final(x)

        if target is None:
            loss = None
        else:
            logits = logits.view(B * T, self.vocab_size)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target)
        return logits, loss

    def generate(self, idx, max_new_tokens,temperature=0.8,top_k=15):
        self.eval()
        for _ in range(max_new_tokens):
            # Le contexte s'allonge à chaque étape
            idx_cond = idx[:, -self.block_size:]
            with torch.no_grad():
                logits, _ = self(idx_cond)
            # La cible est toujours le token immédiatement après ce contexte
            logits = logits[:, -1, :]

            #temperature
            logits = logits / temperature


            #top k
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, -1, None]] = -float('inf') # token en dessous du top k sont mis à -inf pour ne pas être choisis


            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class MusicGeneratorPipeline:
    def __init__(self, path, block_size=256, batch_size=16, n_embd=128, n_head=4, n_layers=4, dropout=0.1):
        # block_size : Longueur du contexte musical vu en une fois
        # n_embd : Taille de la représentation vectorielle de chaque token
        # n_head : Nombre de "têtes" d'attention
        # n_layers : Nombre de couches empilées (profondeur du réseau)
        
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f" CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU ")
            
        self.dataset = MusicDataset(path, block_size, batch_size)
        self.model = MusicGEN(
            self.dataset.vocab_size, 
            block_size, 
            n_embd, 
            n_head, 
            n_layers, 
            self.device,
            dropout=dropout
        ).to(self.device)


    def train_model(self, epochs=50000, learning_rate=6e-4, min_lr=6e-5, warmup_iters=2000, weight_decay=0.1):
        # configuration  de l'optimiseur
        # on ne met pas de Weight Decay sur les biais et les LayerNorms
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))

        #definition d'un lr qui varie 
        def get_lr(it):
            if it < warmup_iters: # warmup 
                return learning_rate * it / warmup_iters 
            if it > epochs:
                return min_lr
            decay_ratio = (it - warmup_iters) / (epochs - warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # cosine decay : le lr diminue en suivant une courbe cosinus
            return min_lr + coeff * (learning_rate - min_lr)

        #  validation loss
        @torch.no_grad()
        def estimate_val_loss(num_batches=10):
            self.model.eval()
            val_losses = []
            for _ in range(num_batches):
                Xv, Yv = self.dataset.get_batch('val', self.device)
                autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
                    logits, loss = self.model(Xv, Yv)
                val_losses.append(loss.item())
            self.model.train()
            return sum(val_losses) / len(val_losses)
        
        self.model.train()
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # MAJ du lr
            lr = get_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            Xb, Yb = self.dataset.get_batch('train', self.device)
            
            # Forward pass
            # CUDA seulement
            autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16): # bfloat16 est un format de nombre flottant qui permet de réduire la taille des données et d'accelerer les calculs
                logits, loss = self.model(Xb, Yb)
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # evite les gradients trop grands
            
            optimizer.step()
            
            if epoch % 500 == 0:
                val_loss = estimate_val_loss(num_batches=10)
                print(f"Step {epoch:5d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.4e}")
                
                # Sauvegarde du meilleur modèle
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model("output/music_model_best.pth")
                    print(f"  -> meilleur model sauvegardé (Val Loss: {val_loss:.4f})")
                
        print(f"Entraînement terminé - final Loss: {loss.item():.4f} - final validation loss :{val_loss:.4f} best Val Loss: {best_val_loss:.4f}")

    def save_model(self, filepath="output/music_model.pth"):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        print(f"modèle sauvegardé dans : {filepath}")

    def load_model(self, filepath="output/music_model.pth"):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device, weights_only=True))
        print(f"modèle chargé depuis : {filepath}")


    def generate_music(self, start_tokens, max_new_tokens=500, output_file="output/musique.mid", temperature=0.8, top_k=15):
        token_ids = [self.dataset.tokenizer[t] for t in start_tokens]
    
        # context tensor 
        context = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        gen_ids = self.model.generate(context, max_new_tokens, temperature=temperature, top_k=top_k)
        ids_list = gen_ids[0].tolist()

        # Creation d'un objet TokSequence avec notre liste
        seq_genere = TokSequence(ids=ids_list)
        self.dataset.tokenizer.complete_sequence(seq_genere)
        
        # conversion en objet symusic.Score
        score_genere = self.dataset.tokenizer.decode([seq_genere])
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        score_genere.dump_midi(output_file)

if __name__ == "__main__":
    # hyperparametres du modele
    block_size = 512   # lonugueur du contexte musical lu d'un coup
    n_embd = 256        # dimenson des vecteurs
    n_head = 8         # nombre de têtes d'attention
    n_layer = 8       # nombre de couches
    dropout = 0.1     # taux de dropout 

    # hyperparametres d'entrainement
    batch_size = 128         # taille des batchs
    max_iters = 55000      # nombre d'étapes d'entrainement
    learning_rate = 6e-4     # learning rate maximal
    min_lr = 3e-5            # learning rate minimal 
    warmup_iters = 2000      # nombre d'étapes d'echauffement
    weight_decay = 0.1       # penalisation des poids trop grands
    
    # Pipeline
    torch.cuda.empty_cache()
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(SCRIPT_DIR, "data")
    pipeline = MusicGeneratorPipeline(path=data_path, block_size=block_size, batch_size=batch_size, n_embd=n_embd, n_head=n_head, n_layers=n_layer, dropout=dropout)
    
    # Train
    #pipeline.train_model(epochs=max_iters, learning_rate=learning_rate, min_lr=min_lr, warmup_iters=warmup_iters, weight_decay=weight_decay)
    
    #sauvegarde des poids
    #pipeline.save_model("output/music_model_maximus.pth")
    
    #loqd
    pipeline.load_model("output/music_model_best.pth")
    
    #generation
    amorce_silence = ["Bar_None", "Position_16"]
    pipeline.generate_music(amorce_silence, max_new_tokens=1000, output_file="output/musique_maximus_topk_temp.mid", temperature=0.8, top_k=15)

