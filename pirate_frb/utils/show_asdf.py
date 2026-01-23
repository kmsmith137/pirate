"""Utility for displaying ASDF file YAML headers."""


def show_asdf(f, out=None):
    """Print the YAML header of an ASDF file (everything before the '...' line).
    
    ASDF files have a YAML header followed by binary data blocks. The YAML
    header ends with a line containing exactly '...'. This function reads
    and prints everything up to and including that line.
    
    Args:
        f: Either a filename (str) or a file-like object opened in binary mode.
        out: Output file-like object for printing (default: sys.stdout).
    """
    import sys
    
    if out is None:
        out = sys.stdout
    
    # Handle both filename and file-like object
    if isinstance(f, str):
        with open(f, 'rb') as fp:
            _show_asdf_impl(fp, out)
    else:
        _show_asdf_impl(f, out)


def _show_asdf_impl(fp, out):
    """Implementation of show_asdf that reads from an open file object."""
    for line in fp:
        # Decode bytes to string, handling potential encoding issues
        try:
            line_str = line.decode('utf-8')
        except UnicodeDecodeError:
            # If we hit binary data before finding '...', stop
            break
        
        # Print the line (rstrip to remove trailing newline, print adds it back)
        out.write(line_str)
        
        # Check if this is the end-of-document marker
        if line_str.rstrip('\r\n') == '...':
            break
