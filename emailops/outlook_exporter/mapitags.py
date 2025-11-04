"""
MAPI property tag constants used via PropertyAccessor.GetProperty.
"""
# Conversation-related
PR_CONVERSATION_ID = "http://schemas.microsoft.com/mapi/proptag/0x30130102"
PR_CONVERSATION_INDEX = "http://schemas.microsoft.com/mapi/proptag/0x00710102"
PR_CONVERSATION_TOPIC = "http://schemas.microsoft.com/mapi/proptag/0x0070001F"
PR_IN_REPLY_TO_ID = "http://schemas.microsoft.com/mapi/proptag/0x1042001F"
PR_REFERENCES = "http://schemas.microsoft.com/mapi/proptag/0x1039001F"
PR_MESSAGE_ID = "http://schemas.microsoft.com/mapi/proptag/0x1035001F"

# Attachments
PR_ATTACH_CONTENT_ID_W = "http://schemas.microsoft.com/mapi/proptag/0x3712001F"
PR_ATTACH_CONTENT_ID_A = "http://schemas.microsoft.com/mapi/proptag/0x3712001E"
PR_ATTACH_FLAGS        = "http://schemas.microsoft.com/mapi/proptag/0x37140003"
PR_ATTACHMENT_HIDDEN   = "http://schemas.microsoft.com/mapi/proptag/0x7FFE000B"
PR_ATTACH_DATA_BIN     = "http://schemas.microsoft.com/mapi/proptag/0x37010102"

# Item properties
PR_ENTRYID                 = "http://schemas.microsoft.com/mapi/proptag/0x0FFF0102"
PR_CLIENT_SUBMIT_TIME      = "http://schemas.microsoft.com/mapi/proptag/0x00390040"
PR_MESSAGE_DELIVERY_TIME   = "http://schemas.microsoft.com/mapi/proptag/0x0E060040"
PR_LAST_MODIFICATION_TIME  = "http://schemas.microsoft.com/mapi/proptag/0x30080040"

# Attachment last-modified timestamp tag (runtime code may fall back to item timestamp)
PR_ATTACH_LAST_MODIFICATION_TIME = PR_LAST_MODIFICATION_TIME

# Address resolution
PR_SMTP_ADDRESS = "http://schemas.microsoft.com/mapi/proptag/0x39FE001E"
