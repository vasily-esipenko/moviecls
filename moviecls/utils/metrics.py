import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def train_epoch(model, dataloader, optimizer, criterion, text_embeddings, device):
    model.train()
    total_loss = 0

    from moviecls.utils.embeddings import get_text_embeddings_batch

    for batch in tqdm(dataloader, desc="Обучение"):
        images = batch['image'].to(device)
        text_emb = get_text_embeddings_batch(batch, text_embeddings, device)

        visual_emb = model(images)

        target = torch.ones(images.size(0)).to(
            device)  # Цель - косинусное сходство = 1
        loss = criterion(visual_emb, text_emb, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, text_embeddings, device, precision_k=10):
    model.eval()
    total_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    sum_reciprocal_ranks = 0
    sum_precision_at_k = 0

    from moviecls.utils.embeddings import get_text_embeddings_batch

    all_movie_ids = list(text_embeddings.keys())
    all_text_emb = np.array([text_embeddings[mid] for mid in all_movie_ids])

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Валидация"):
            images = batch['image'].to(device)
            text_emb = get_text_embeddings_batch(
                batch, text_embeddings, device)

            visual_emb = model(images)

            target = torch.ones(images.size(0)).to(device)
            loss = criterion(visual_emb, text_emb, target)
            total_loss += loss.item()

            visual_emb_np = visual_emb.cpu().numpy()

            similarities = cosine_similarity(visual_emb_np, all_text_emb)

            topk_indices = np.argsort(-similarities, axis=1)[:, :precision_k]
            top5_indices = topk_indices[:, :5]

            for i, movie_id in enumerate(batch['movie_id']):
                if movie_id == all_movie_ids[top5_indices[i, 0]]:
                    correct_top1 += 1

                if movie_id in [all_movie_ids[idx] for idx in top5_indices[i]]:
                    correct_top5 += 1

                correct_positions = np.where(np.array(all_movie_ids)[
                                             topk_indices[i]] == movie_id)[0]
                if len(correct_positions) > 0:
                    rank = correct_positions[0] + 1
                    sum_reciprocal_ranks += 1.0 / rank

                relevant_in_topk = 1 if movie_id in [
                    all_movie_ids[idx] for idx in topk_indices[i]] else 0
                sum_precision_at_k += relevant_in_topk / precision_k

            total += images.size(0)

    avg_loss = total_loss / len(dataloader)
    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total

    mrr = sum_reciprocal_ranks / total
    precision_at_k = sum_precision_at_k / total

    return avg_loss, top1_acc, top5_acc, mrr, precision_at_k


def find_movie_by_frame(frame_path, model, text_embeddings, transform, device, top_k=5):
    from PIL import Image
    model.eval()

    all_movie_ids = list(text_embeddings.keys())
    all_text_emb = np.array([text_embeddings[mid] for mid in all_movie_ids])

    image = Image.open(frame_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        visual_emb = model(image_tensor).cpu().numpy()

    similarities = cosine_similarity(visual_emb, all_text_emb)[0]

    top_k_indices = np.argsort(-similarities)[:top_k]

    results = []
    for idx in top_k_indices:
        movie_id = all_movie_ids[idx]
        similarity = similarities[idx]
        results.append({
            'movie_id': movie_id,
            'similarity': float(similarity),
        })

    return results
