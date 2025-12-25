"""
Utility functions for post-processing YAML output.

These functions work around limitations in yaml-cpp's comment handling.
"""

import re


def indent_dedispersion_plan_comments(yaml_str):
    """
    Hack to indent multiline comments in YAML output from DedispersionPlan::to_yaml().
    
    yaml-cpp doesn't support indented comments, so we post-process the output to indent
    multiline comments that start with "# At tree_index=..." by 4 spaces.
    """
    lines = yaml_str.split('\n')
    result = []
    in_block = False
    
    for line in lines:
        if line.startswith('# At tree_index='):
            # Start of a multiline comment block
            in_block = True
            result.append('    ' + line)
        elif in_block and line.startswith('#'):
            # Continuation of the multiline comment block
            result.append('    ' + line)
        else:
            # Not in a comment block, or end of block
            in_block = False
            result.append(line)
    
    return '\n'.join(result)


def align_inline_comments(yaml_str):
    """
    Horizontally align inline comments within blocks of consecutive lines.
    
    A "block" is a sequence of consecutive lines where each line contains both
    non-comment text AND an inline comment. Empty lines, lines that are entirely
    comments (no preceding non-whitespace text), or lines without comments start a new block.
    
    Within each block, comments are padded to align at the same column.
    """
    lines = yaml_str.split('\n')
    result = []
    
    # Pattern to match a line with non-whitespace, non-comment text followed by a comment.
    # The text part must contain at least one non-whitespace character before the comment.
    # Captures: (text before comment, comment including #)
    inline_comment_pattern = re.compile(r'^(\s*\S[^#]*?)\s*(#.*)$')
    
    i = 0
    while i < len(lines):
        # Try to collect a block of lines with inline comments
        block = []
        block_indices = []
        
        while i < len(lines):
            line = lines[i]
            match = inline_comment_pattern.match(line)
            
            if match:
                # Line has non-comment text + inline comment
                text_part = match.group(1).rstrip()
                comment_part = match.group(2)
                block.append((text_part, comment_part))
                block_indices.append(i)
                i += 1
            else:
                # Line doesn't match pattern - ends the block
                break
        
        if len(block) > 0:
            # Align comments in this block
            max_text_len = max(len(text) for text, _ in block)
            
            for idx, (text, comment) in zip(block_indices, block):
                # Pad text to align comments, with 2 spaces before comment
                padded_line = text.ljust(max_text_len) + '  ' + comment
                result.append(padded_line)
        
        # Add the non-matching line (if any) that ended the block
        if i < len(lines):
            result.append(lines[i])
            i += 1
    
    return '\n'.join(result)

