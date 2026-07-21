"""Single source of truth for storage keys and the citation grammar."""
import re

# ONE definition shared by the grounding gate (agent) and the renderer (bot).
# 【】 has no meaning in Markdown and never occurs in the corpus texts.
CITE_RE = re.compile(r"【([^【】#]+)#(\d+)】")


def thread_id_for(tenant_id, chat_id):
    """Memory scope = the CHAT (private -> personal, group -> shared)."""
    return f"{tenant_id}:{chat_id}"
