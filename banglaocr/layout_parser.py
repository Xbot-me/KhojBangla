from ocr_engines import TextBlock

def filter_ads(blocks: list[TextBlock]) -> list[TextBlock]:
    """
    Filter out blocks that are likely advertisements based on geometric and text heuristics.
    """
    filtered = []
    for block in blocks:
        # Calculate block-level stats
        total_words = sum(len(p.words) for p in block.paragraphs)
        if total_words == 0:
            continue
            
        # Example heuristic: if a block is very large but has very few words, it's likely a graphic ad.
        # block area: block.bbox.w * block.bbox.h
        area = block.bbox.w * block.bbox.h
        words_per_area = total_words / max(area, 1)
        
        # Another heuristic: ads often have very large text or very small text.
        # A simple check for now: just exclude blocks with extreme word sparseness.
        if words_per_area < 0.0001:  # arbitrary threshold for now
            # Skip this block as an ad
            continue
            
        # We can also check bounding box aspect ratios or specific text patterns (e.g. phone numbers)
        # For this prototype, we'll keep it simple.
        filtered.append(block)
        
    return filtered

def group_articles(blocks: list[TextBlock]) -> list[list[TextBlock]]:
    """
    Group blocks into cohesive articles based on column alignment and proximity.
    For now, return each block as its own article group, as Vision's block detection 
    is usually fairly good at isolating distinct articles or columns.
    """
    # A more advanced implementation would merge blocks that are vertically adjacent
    # and have similar widths.
    return [[b] for b in blocks]
