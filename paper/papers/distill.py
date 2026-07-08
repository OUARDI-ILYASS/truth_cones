import argparse
import json
import os
from pathlib import Path
import sys

def log(msg):
    """Print with immediate flush for background tasks."""
    print(msg, flush=True)
    sys.stdout.flush()



def repair_json(text: str) -> dict:
    """Attempt to parse JSON with common LLM output repairs.

    Handles: trailing commas, unescaped newlines in strings, markdown fences.
    """
    import re

    # Strip markdown fences
    if '```' in text:
        text = re.sub(r'```(?:json)?\s*', '', text)

    # Find the JSON object
    start = text.find('{')
    end = text.rfind('}') + 1
    if start < 0 or end <= start:
        return None
    text = text[start:end]

    # First try: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Remove trailing commas before } or ]
    cleaned = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Replace unescaped control characters in strings
    cleaned = re.sub(r'[\x00-\x1f]', ' ', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    return None


def extract_text_from_pdf(path: str) -> str:
    """Extract text from PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("Install PyMuPDF: pip install pymupdf")
        sys.exit(1)
    
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n\n"
    doc.close()
    return text



def extract_text(path: str) -> str:
    """Extract text from any supported format."""
    ext = Path(path).suffix.lower()
    if ext == '.pdf':
        return extract_text_from_pdf(path)
    elif ext in ('.md', '.txt', '.text'):
        return Path(path).read_text(encoding='utf-8', errors='replace')
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: .pdf, .docx, .md, .txt, .mhtml, .pages, .html")

def count_tokens_approx(text: str) -> int:
    """Approximate token count (1 token ≈ 4 chars for English)."""
    return len(text) // 4

def semantic_chunk(text: str, max_tokens: int = 4000, overlap_tokens: int = 500, total_chars: int = 0) -> list[dict]:
    """Split text into overlapping chunks on paragraph boundaries.

    Args:
        text: Full text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap between chunks
        total_chars: Total characters in document (for position estimation)

    Returns:
        List of chunks with provenance metadata
    """
    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4

    paragraphs = text.split('\n\n')
    chunks = []
    current = ""
    chunk_id = 0
    char_position = 0

    for para in paragraphs:
        if count_tokens_approx(current + para) > max_tokens and current:
            # Estimate page number (assume 2500 chars/page)
            approx_page = char_position // 2500 + 1
            chunks.append({
                "id": chunk_id,
                "text": current.strip(),
                "tokens": count_tokens_approx(current),
                "char_start": char_position,
                "char_end": char_position + len(current),
                "approx_page": approx_page,
            })
            chunk_id += 1
            # Overlap: keep the tail of the previous chunk
            current = current[-overlap_chars:] + "\n\n" + para
        else:
            current += "\n\n" + para if current else para

        char_position += len(para) + 2  # +2 for newlines

    if current.strip():
        approx_page = char_position // 2500 + 1
        chunks.append({
            "id": chunk_id,
            "text": current.strip(),
            "tokens": count_tokens_approx(current),
            "char_start": char_position,
            "char_end": char_position + len(current),
            "approx_page": approx_page,
        })

    return chunks



def distill_file(
    filepath: str,
    output_dir: str,
    output_mode: str = "knowledge-map",
    max_concurrent: int = 10,
    use_hierarchical: bool = True,
    use_v2: bool = False,
    use_multi_turn: bool = False,
    topic_index: dict = None,
) -> dict:
    """Run the full distillation pipeline on a single file."""
    
    log(f"\n{'='*60}")
    log(f"Distilling: {filepath}")
    log(f"Output mode: {output_mode}")
    log(f"{'='*60}")
    
    # Extract text
    log(f"\n📖 Extracting text...")
    text = extract_text(filepath)
    total_tokens = count_tokens_approx(text)
    log(f"   {total_tokens:,} tokens ({total_tokens // 500} pages approx)")
    
    # Chunk
    log(f"\n✂️  Chunking...")
    chunks = semantic_chunk(text, total_chars=len(text))
    log(f"   {len(chunks)} chunks")

    p3_path = os.path.join(output_dir, f"chunks.json")
    with open(p3_path, 'w') as f:
        json.dump(chunks, f, indent=2)

    return chunks
    
    # # Pass 1: Haiku extraction
    # if use_v2 and V2_AVAILABLE:
    #     log(f"\n🐝 Pass 1 (V2): Adaptive probe extraction ({len(chunks)} parallel)...")
    #     extractions = await pass1_extract_v2(client, chunks, max_concurrent)
    # else:
    #     log(f"\n🐝 Pass 1: Haiku army ({len(chunks)} parallel extractions)...")
    #     extractions = await pass1_extract(client, chunks, max_concurrent)
    # p1_input = sum(e["input_tokens"] for e in extractions)
    # p1_output = sum(e["output_tokens"] for e in extractions)
    # p1_cost = (p1_input * 0.80 + p1_output * 4.00) / 1_000_000
    # errors = sum(1 for e in extractions if "error" in e.get("extraction", {}))
    # log(f"   Done. {len(extractions) - errors} succeeded, {errors} errors")
    # log(f"   Cost: ${p1_cost:.4f} ({p1_input:,} in + {p1_output:,} out)")
    
    # # Save Pass 1 output
    # p1_path = os.path.join(output_dir, f"{filename}_pass1_extractions.json")
    # with open(p1_path, 'w') as f:
    #     json.dump(extractions, f, indent=2)
    # log(f"   Saved: {p1_path}")
    
    # if output_mode == "summary":
    #     # Just return the concatenated summaries
    #     summaries = [e["extraction"].get("summary", "") for e in extractions if "error" not in e.get("extraction", {})]
    #     result = {"summaries": summaries, "pass1_cost": p1_cost}
    #     result_path = os.path.join(output_dir, f"{filename}_summary.json")
    #     with open(result_path, 'w') as f:
    #         json.dump(result, f, indent=2)
    #     print(f"\n✅ Summary saved: {result_path}")
    #     print(f"   Total cost: ${p1_cost:.4f}")
    #     return result
    
    # # Pass 2: Sonnet synthesis
    # if use_multi_turn:
    #     log(f"\n🧠 Pass 2: Sonnet synthesis (multi-turn lookup enabled)...")
    # else:
    #     log(f"\n🧠 Pass 2: Sonnet synthesis...")
    # synthesis = await pass2_synthesize(
    #     client,
    #     extractions,
    #     use_hierarchical=use_hierarchical,
    #     use_multi_turn=use_multi_turn,
    #     topic_index=topic_index,
    #     source_name=filename,  # Use filename as source filter
    # )

    # # Calculate cost (use actual_cost from hierarchical merge if available)
    # if "actual_cost" in synthesis:
    #     p2_cost = synthesis["actual_cost"]
    # else:
    #     p2_cost = (synthesis["input_tokens"] * 3.00 + synthesis["output_tokens"] * 15.00) / 1_000_000

    # log(f"   Cost: ${p2_cost:.4f}")
    
    # # Save Pass 2 output
    # p2_path = os.path.join(output_dir, f"{filename}_knowledge_map.json")
    # with open(p2_path, 'w') as f:
    #     json.dump(synthesis["knowledge_map"], f, indent=2)
    # log(f"   Saved: {p2_path}")
    
    # if output_mode == "knowledge-map":
    #     total_cost = p1_cost + p2_cost
    #     print(f"\n✅ Knowledge map saved: {p2_path}")
    #     print(f"   Total cost: ${total_cost:.4f}")
    #     return {"knowledge_map": synthesis["knowledge_map"], "total_cost": total_cost}
    
    # # Pass 3: Skill draft or Recipe generation
    # if use_v2 and V2_AVAILABLE:
    #     log(f"\n📝 Pass 3 (V2): Recipe generation...")
    #     recipes = await generate_recipes(client, synthesis["knowledge_map"])
    #     p3_cost = recipes.get("_generation_metadata", {}).get("cost", 0.05)

    #     # Save recipe directory
    #     recipe_dir = os.path.join(output_dir, f"{filename}_recipes")
    #     os.makedirs(recipe_dir, exist_ok=True)

    #     # Save full recipe JSON
    #     recipe_path = os.path.join(recipe_dir, "recipes.json")
    #     with open(recipe_path, 'w') as f:
    #         json.dump(recipes, f, indent=2)

    #     # Save individual recipe files for easy browsing
    #     for recipe in recipes.get("recipes", []):
    #         rid = recipe.get("recipe_id", f"recipe-{recipes.get('recipes', []).index(recipe)}")
    #         individual_path = os.path.join(recipe_dir, f"{rid}.json")
    #         with open(individual_path, 'w') as f:
    #             json.dump(recipe, f, indent=2)

    #     # Save glossary
    #     if recipes.get("glossary"):
    #         glossary_path = os.path.join(recipe_dir, "glossary.json")
    #         with open(glossary_path, 'w') as f:
    #             json.dump(recipes["glossary"], f, indent=2)

    #     # Save the knowledge map alongside
    #     km_path = os.path.join(recipe_dir, "knowledge_map.json")
    #     with open(km_path, 'w') as f:
    #         json.dump(synthesis["knowledge_map"], f, indent=2)

    #     total_cost = p1_cost + p2_cost + p3_cost
    #     recipe_count = len(recipes.get("recipes", []))
    #     log(f"\n✅ Recipe directory saved: {recipe_dir}")
    #     log(f"   {recipe_count} recipes generated")
    #     log(f"   Total cost: ${total_cost:.4f}")

    #     return {"recipes": recipes, "knowledge_map": synthesis["knowledge_map"], "total_cost": total_cost}
    # else:
    #     log(f"\n📝 Pass 3: Skill draft generation...")
    #     skill_md = await pass3_skill_draft(client, synthesis["knowledge_map"])

    #     # Save skill draft
    #     p3_path = os.path.join(output_dir, f"{filename}_SKILL.md")
    #     with open(p3_path, 'w') as f:
    #         f.write(skill_md)

    #     total_cost = p1_cost + p2_cost + 0.05  # Approximate Pass 3 cost
    #     log(f"\n✅ Skill draft saved: {p3_path}")
    #     log(f"   Total cost: ${total_cost:.4f}")

    #     return {"skill_draft": skill_md, "knowledge_map": synthesis["knowledge_map"], "total_cost": total_cost}



def main():
    parser = argparse.ArgumentParser(description="Corpus Distillation Pipeline")
    parser.add_argument("input", help="File path or directory of files to distill")
    parser.add_argument("--output-mode", choices=["summary", "knowledge-map", "skill-draft"],
                        default="knowledge-map", help="Output mode (default: knowledge-map)")
    parser.add_argument("--output-dir", default="./", help="Output directory")
    parser.add_argument("--hierarchical", action="store_true", default=True,
                        help="Use hierarchical binary merge for Phase 2 (recommended, default: True)")
    parser.add_argument("--no-hierarchical", action="store_false", dest="hierarchical",
                        help="Use legacy flat merge (may hit context limits)")

    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        distill_file(
            str(input_path),
            args.output_dir,
            args.output_mode,
            use_hierarchical=args.hierarchical,
        )



if __name__ == "__main__":
    main()

