Option Explicit

'====================================================================
' Module: ConvTxt_Builder_AllInOne
' (Streaming, nested messages, multi-folder; attachments + CSV log;
'  manifest.json + conversations_index.jsonl with idempotent re-runs)
'
' Highlights:
'   - Streams processing (files appear immediately; no giant pre-pass)
'   - Items.Restrict to IPM.Note (fast, safe across stores)
'   - De-dupe by Internet Message-ID; stable same-run signature
'   - Non-destructive HTML pre-pass; robust quote/history cutter
'   - Minimum-keep guard + signature-only fallback
'   - Tail-only disclaimer trimming
'   - HtmlToText preserves bullets + quoted lines + breaks
'   - Unicode & numeric entity normalization
'   - Linear-time whitespace normalize; faster JoinUpTo w/ bounds clamp
'   - Cached EffectiveLocalTime; precomputed-time quicksort
'   - Atomic writes (UTF-8 via ADODB.Stream) + FSO fallback (Unicode)
'   - Depth-limited nested .msg/.eml; multi-pass folder scan
'   - Optional trace to Immediate window; responsive (DoEvents)
'   - Includes original outbound from Sent Items across all stores
'   - Preserves attachment file extensions when trimming to path budget
'   - Reserves folder-name budget for attachments to reduce truncation
'   - manifest.json (canonical v1: manifest_version=1, folder, subject_label,
'     message_count, started_at_utc, ended_at_utc, attachment_count,
'     paths{Conversation.txt + attachments/}, sha256_conversation)
'     + global conversations_index.jsonl with idempotent appends
'     (via .indexed.sha256 marker)
'   - Output folder picker (FileDialog FolderPicker) per run
'   - Drafts excluded (unsent mail items skipped)
'====================================================================

'=== CONFIG =====================================================================
Private Const OUT_ROOT As String = "C:\Users\ASUS\Desktop\Outlook"   ' base output folder (fallback is %LOCALAPPDATA%)
Private Const FLAG_ADDR As String = "hagop.ghazarian@chalhoub.com"   ' TML flag address
Private Const MAX_PATH_BUDGET As Long = 240                          ' keep under MAX_PATH (~260)

' Keep at least this many characters available for attachment filenames
' (folder naming will leave this headroom for \attachments\<filename>)
Private Const RESERVED_ATTACHMENT_FILENAME_CHARS As Long = 64

' Show a MsgBox every N conversations (0 = only final)
Private Const SHOW_SUMMARY_EVERY_N As Long = 0

' OPTIONAL: print progress to the Immediate window (Ctrl+G)
Private Const TRACE_PROGRESS As Boolean = False

'=== SUBFOLDERS & NESTED ========================================================
Private Const PROCESS_SUBFOLDERS As Boolean = True

Private Const PROCESS_NESTED_DEPTH_MAX As Long = 3     ' recursion depth for nested attached messages
Private Const NESTED_FOLDER_SCAN_PASSES As Long = 10   ' passes scanning attachments dir for saved .msg/.eml
Private Const NESTED_SUFFIX As String = "_conv_nested" ' suffix for nested conversation text files

' Avoid enum name/case issues; Outlook OlAttachmentType: 5 = embedded item
Private Const OL_ATTACHMENTTYPE_EMBEDDED As Long = 5

'=== ATTACHMENTS ================================================================
Private Const ATT_SUBDIR As String = "attachments"
Private Const INCLUDE_INLINE_UPLOADED As Boolean = True
Private Const MAX_SIGNATURE_IMAGE_BYTES As Long = 12000
Private Const MIN_INLINE_UPLOAD_BYTES As Long = 15000

'=== MAPI PROPS ================================================================
Private Const PR_ATTACHMENT_HIDDEN As String = "http://schemas.microsoft.com/mapi/proptag/0x7FFE000B"
Private Const PR_LAST_MODIFICATION_TIME As String = "http://schemas.microsoft.com/mapi/proptag/0x30080040"
Private Const PR_CREATION_TIME As String = "http://schemas.microsoft.com/mapi/proptag/0x30070040"
Private Const PR_ATTACH_CONTENT_ID As String = "http://schemas.microsoft.com/mapi/proptag/0x3712001E"
Private Const PR_ATTACH_CONTENT_DISPOSITION As String = "http://schemas.microsoft.com/mapi/proptag/0x3716001E"
Private Const PR_ATTACH_MIME_TAG As String = "http://schemas.microsoft.com/mapi/proptag/0x370E001E"
Private Const PR_INTERNET_MESSAGE_ID_A As String = "http://schemas.microsoft.com/mapi/proptag/0x1035001E"
Private Const PR_INTERNET_MESSAGE_ID_W As String = "http://schemas.microsoft.com/mapi/proptag/0x1035001F"

'=== ENTRY POINTS ==============================================================

' Selection-aware version — processes ALL selected MailItems (and expands ConversationHeaders)
Public Sub BuildConversationTxt_FromSelection()
    On Error GoTo EH
    Dim app As Outlook.Application: Set app = Application

    Dim selMails As Collection
    Set selMails = GetSelectedMailItems(app)

    If selMails Is Nothing Or selMails.count = 0 Then
        MsgBox "Select one or more mail items (or open a message) first.", vbExclamation, "Conversation.txt (selection)"
        Exit Sub
    End If

    ' Single in-run de-dupe across the whole selection
    Dim processed As Object: Set processed = CreateObject("Scripting.Dictionary"): processed.CompareMode = 1

    Dim processedCount As Long, skippedCount As Long
    Dim runMsgs As New Collection

    Dim i As Long
    For i = 1 To selMails.count
        Dim mi As Outlook.MailItem: Set mi = selMails(i)

        Dim result As String
        result = ProcessConversationIfNew(mi, processed)

        If Left$(result, 5) = "SKIP:" Then
            skippedCount = skippedCount + 1
        Else
            processedCount = processedCount + 1
            runMsgs.Add result
        End If

        Trace "sel -> " & result
        DoEvents
    Next i

    ' Final summary
    Dim finalReport As String
    finalReport = "Selection run complete." & vbCrLf & _
                  "Conversations processed: " & processedCount & vbCrLf & _
                  "Skipped (already processed in this run / drafts / errors): " & skippedCount & vbCrLf & _
                  String$(60, "-") & vbCrLf

    Dim showN As Long: showN = 10
    Dim j As Long
    For j = 1 To runMsgs.count
        If j > showN Then finalReport = finalReport & "...": Exit For
        finalReport = finalReport & CStr(runMsgs(j)) & vbCrLf
    Next j

    MsgBox finalReport, vbInformation, "Conversation.txt (selection)"
    Exit Sub
EH:
    MsgBox "Error (" & Err.Number & "): " & Err.Description, vbCritical, "Conversation.txt (selection)"
End Sub

' Folder-based entry point
Public Sub BuildConversationTxt_FromSelectedFolders()
    On Error GoTo EH

    Dim app As Outlook.Application: Set app = Application

    ' 1) Pick one or more folders (PickFolder loop; Cancel to finish)
    Dim folders As Collection: Set folders = PickMultipleFolders(app)
    If folders Is Nothing Or folders.count = 0 Then
        MsgBox "No folders selected.", vbExclamation, "Conversation.txt (folders)"
        Exit Sub
    End If

    ' 2) Stream processing per folder (output appears immediately)
    Dim processed As Object: Set processed = CreateObject("Scripting.Dictionary"): processed.CompareMode = 1
    Dim processedCount As Long, skippedCount As Long
    Dim runMsgs As New Collection

    Dim i As Long
    For i = 1 To folders.count
        ProcessFolderStreaming folders(i), PROCESS_SUBFOLDERS, processed, processedCount, skippedCount, runMsgs
    Next i

    ' 3) Final summary
    Dim finalReport As String
    finalReport = "Folders run complete." & vbCrLf & _
                  "Conversations processed: " & processedCount & vbCrLf & _
                  "Skipped (already processed in this run / drafts / errors): " & skippedCount & vbCrLf & _
                  String$(60, "-") & vbCrLf

    Dim showN As Long: showN = 10
    Dim j As Long
    For j = 1 To runMsgs.count
        If j > showN Then finalReport = finalReport & "...": Exit For
        finalReport = finalReport & CStr(runMsgs(j)) & vbCrLf
    Next j

    MsgBox finalReport, vbInformation, "Conversation.txt (folders)"
    Exit Sub
EH:
    MsgBox "Error (" & Err.Number & "): " & Err.Description, vbCritical, "Conversation.txt (folders)"
End Sub

'=== STREAMING FOLDER WALK ======================================================

Private Sub ProcessFolderStreaming(ByVal root As Outlook.MAPIFolder, ByVal includeSub As Boolean, _
                                   ByVal processed As Object, ByRef processedCount As Long, _
                                   ByRef skippedCount As Long, ByRef runMsgs As Collection)
    On Error GoTo EH
    If root Is Nothing Then Exit Sub

    ' Safely get FolderPath for tracing
    Dim rootPath As String
    On Error Resume Next
    rootPath = root.folderPath
    If Err.Number <> 0 Then
        rootPath = "(Unavailable Folder)"
        Err.Clear
    End If
    On Error GoTo EH

    Trace "Folder: " & rootPath

    Dim itms As Outlook.items
    ' Localised OERN for potentially failing property access (e.g., offline store)
    On Error Resume Next
    Set itms = root.items
    If Err.Number <> 0 Then
        Trace "  ERROR accessing items in " & rootPath & ": (" & Err.Number & ") " & Err.Description
        Err.Clear
        ' If we can't access items, we skip processing this folder's content but still try subfolders.
        On Error GoTo EH
        GoTo ProcessSubfolders
    End If
    On Error GoTo EH

    If Not itms Is Nothing Then
        ' Restrict to mail items for speed
        Dim mails As Outlook.items

        On Error Resume Next
        Set mails = itms.Restrict("[MessageClass] = 'IPM.Note'")
        If Err.Number <> 0 Or mails Is Nothing Then
            If Err.Number <> 0 Then
                Trace "  WARN: Items.Restrict failed for " & rootPath & ". Falling back to full item scan. (" & Err.Number & ")"
                Err.Clear
            End If
            Set mails = itms    ' safe fallback
        End If
        On Error GoTo EH

        Dim obj As Object
        For Each obj In mails
            On Error Resume Next

            If TypeName(obj) = "MailItem" Then
                Dim mi As Outlook.MailItem
                Set mi = obj

                Dim result As String
                result = ProcessConversationIfNew(mi, processed)

                If Left$(result, 5) = "SKIP:" Then
                    skippedCount = skippedCount + 1
                Else
                    processedCount = processedCount + 1
                    runMsgs.Add result
                    If SHOW_SUMMARY_EVERY_N > 0 Then
                        If (processedCount Mod SHOW_SUMMARY_EVERY_N) = 0 Then
                            MsgBox "Processed so far: " & processedCount & vbCrLf & _
                                   "Skipped (already processed in this run / drafts / errors): " & skippedCount, _
                                   vbInformation, "Progress"
                        End If
                    End If
                End If

                Trace "  -> " & result
                DoEvents
            End If

            If Err.Number <> 0 Then
                Trace "  ERROR processing item in " & rootPath & ": (" & Err.Number & ") " & Err.Description
                Err.Clear
                skippedCount = skippedCount + 1
            End If

            On Error GoTo EH
        Next obj
    End If

ProcessSubfolders:
    If includeSub Then
        Dim subf As Outlook.MAPIFolder
        Dim rootFolders As Outlook.folders

        On Error Resume Next
        Set rootFolders = root.folders
        If Err.Number <> 0 Then
            Trace "  ERROR accessing subfolders in: " & rootPath & " (" & Err.Number & "): " & Err.Description
            Err.Clear
        Else
            On Error GoTo EH
            If Not rootFolders Is Nothing Then
                For Each subf In rootFolders
                    ProcessFolderStreaming subf, True, processed, processedCount, skippedCount, runMsgs
                Next subf
            End If
        End If
    End If

    Exit Sub

EH:
    Dim currentPath As String
    On Error Resume Next
    currentPath = root.folderPath
    If Err.Number <> 0 Then currentPath = "(Unknown Path)"
    Err.Clear
    On Error GoTo 0

    Trace "Unexpected ERROR in ProcessFolderStreaming for " & currentPath & " (" & Err.Number & "): " & Err.Description
    ' Swallow and let caller continue with other folders
End Sub

Private Sub Trace(ByVal s As String)
    If TRACE_PROGRESS Then Debug.Print Format$(Now, "hh:nn:ss"), s
End Sub

'=== SELECTION HELPERS ==========================================================

' OLD helper kept for compatibility (returns only one item)
Public Function GetSelectedMailItem(ByVal app As Outlook.Application) As Outlook.MailItem
    On Error Resume Next
    Dim exp As Outlook.Explorer: Set exp = app.ActiveExplorer
    If Not exp Is Nothing Then
        Dim sel As Outlook.selection: Set sel = exp.selection
        If Not sel Is Nothing And sel.count > 0 Then
            If TypeName(sel.item(1)) = "MailItem" Then Set GetSelectedMailItem = sel.item(1): Exit Function
        End If
    End If
    Dim insp As Outlook.Inspector: Set insp = app.ActiveInspector
    If Not insp Is Nothing Then
        If TypeName(insp.CurrentItem) = "MailItem" Then Set GetSelectedMailItem = insp.CurrentItem
    End If
End Function

' NEW: returns ALL selected MailItems; expands ConversationHeader where possible.
Private Function GetSelectedMailItems(ByVal app As Outlook.Application) As Collection
    On Error Resume Next
    Dim result As New Collection

    Dim exp As Outlook.Explorer: Set exp = app.ActiveExplorer
    If Not exp Is Nothing Then
        Dim sel As Outlook.selection: Set sel = exp.selection
        If Not sel Is Nothing And sel.count > 0 Then
            Dim i As Long
            For i = 1 To sel.count
                Dim obj As Object: Set obj = sel.item(i)
                Select Case TypeName(obj)
                    Case "MailItem"
                        result.Add obj
                    Case "ConversationHeader"
                        ' Best-effort expand of a collapsed conversation header
                        Dim hdr As Object: Set hdr = obj
                        Dim items As Object
                        On Error Resume Next
                        Set items = hdr.GetItems   ' Outlook returns SimpleItems
                        On Error GoTo 0
                        If Not items Is Nothing Then
                            Dim it As Object
                            For Each it In items
                                If TypeName(it) = "MailItem" Then result.Add it
                            Next it
                        End If
                End Select
            Next i
        End If
    End If

    ' If nothing in the Explorer selection, try the open Inspector
    If result.count = 0 Then
        Dim insp As Outlook.Inspector: Set insp = app.ActiveInspector
        If Not insp Is Nothing Then
            If TypeName(insp.CurrentItem) = "MailItem" Then result.Add insp.CurrentItem
        End If
    End If

    If result.count = 0 Then
        Set GetSelectedMailItems = Nothing
    Else
        Set GetSelectedMailItems = result
    End If
End Function

Private Function PickMultipleFolders(ByVal app As Outlook.Application) As Collection
    On Error Resume Next
    Dim picked As New Collection
    Dim seen As Object: Set seen = CreateObject("Scripting.Dictionary"): seen.CompareMode = 1

    Do
        Dim f As Outlook.MAPIFolder
        Set f = app.Session.PickFolder
        If f Is Nothing Then Exit Do
        Dim k As String: k = LCase$(NullToEmpty(f.entryId))
        If Len(k) = 0 Then k = LCase$(NullToEmpty(f.folderPath))
        If Len(k) > 0 Then
            If Not seen.Exists(k) Then
                seen(k) = True
                picked.Add f
            End If
        End If
    Loop

    If picked.count = 0 Then
        Set PickMultipleFolders = Nothing
    Else
        Set PickMultipleFolders = picked
    End If
    On Error GoTo 0
End Function

'=== CONVERSATION ENUM (branching) =============================================

Private Function CollectConversationMails(ByVal seed As Outlook.MailItem) As Collection
    On Error GoTo EH
    Dim conv As Outlook.Conversation: Set conv = seed.GetConversation ' may be Nothing
    Dim result As New Collection
    Dim seen As Object: Set seen = CreateObject("Scripting.Dictionary")
    If conv Is Nothing Then
        AddMailIfNew result, seed, seen
        Set CollectConversationMails = result
        Exit Function
    End If

    Dim roots As Object: Set roots = conv.GetRootItems
    Dim node As Object
    For Each node In roots
        TraverseConversation conv, node, result, seen
    Next node
    Set CollectConversationMails = result
    Exit Function
EH:
    Dim fb As New Collection
    AddMailIfNew fb, seed, CreateObject("Scripting.Dictionary")
    Set CollectConversationMails = fb
End Function

Private Sub TraverseConversation(ByVal conv As Outlook.Conversation, ByVal node As Object, _
                                 ByRef out As Collection, ByRef seen As Object)
    On Error Resume Next
    Dim mi As Outlook.MailItem
    If TypeName(node) = "MailItem" Then Set mi = node
    If Not mi Is Nothing Then AddMailIfNew out, mi, seen
    Dim kids As Object: Set kids = conv.GetChildren(node)
    If Not kids Is Nothing Then
        Dim child As Object
        For Each child In kids
            TraverseConversation conv, child, out, seen
        Next child
    End If
End Sub

'=== MESSAGE-ID + RUN SIGNATURE ================================================

Private Function GetInternetMessageId(ByVal mi As Outlook.MailItem) As String
    On Error Resume Next
    Dim s As String
    s = NullToEmpty(mi.InternetMessageID)
    If Len(s) = 0 Then
        Dim pa As Object
        Set pa = mi.PropertyAccessor
        If Not pa Is Nothing Then
            s = NullToEmpty(pa.GetProperty(PR_INTERNET_MESSAGE_ID_W))
            If Len(s) = 0 Then s = NullToEmpty(pa.GetProperty(PR_INTERNET_MESSAGE_ID_A))
        End If
    End If
    s = Trim$(s)
    If Left$(s, 1) = "<" Then s = mid$(s, 2)
    If Len(s) > 0 And Right$(s, 1) = ">" Then s = Left$(s, Len(s) - 1)
    GetInternetMessageId = LCase$(s)
End Function

Private Function BuildStableIdSignature(ByVal msgs As Collection) As String
    On Error Resume Next
    Dim n As Long: n = msgs.count
    If n = 0 Then
        BuildStableIdSignature = "sig:00000000"
        Exit Function
    End If

    Dim ids() As String
    ReDim ids(1 To n)
    Dim allEmpty As Boolean
    allEmpty = True

    Dim i As Long
    For i = 1 To n
        ids(i) = GetInternetMessageId(msgs(i))
        If Len(ids(i)) > 0 Then allEmpty = False
    Next i

    If allEmpty Then
        ' Fallback: use our robust identity key for every message
        ReDim ids(1 To n)
        For i = 1 To n
            Dim mi As Outlook.MailItem
            Set mi = msgs(i)
            Dim k As String
            k = MessageIdentityKey(mi)
            If Len(k) = 0 Then k = "fb:empty"
            ids(i) = k
        Next i
    End If

    QuickSortStrings ids, 1, UBound(ids)
    BuildStableIdSignature = "sig:" & ShortHash(Join(ids, "|"), 12)
End Function

' Centralised helper function for robust identity key
Private Function MessageIdentityKey(ByVal mi As Outlook.MailItem) As String
    On Error Resume Next
    If mi Is Nothing Then Exit Function

    Dim key As String

    ' 1. Internet Message ID (most stable)
    key = GetInternetMessageId(mi)
    If Len(key) > 0 Then
        MessageIdentityKey = "I:" & key
        Exit Function
    End If

    ' 2. Entry ID (stable within a store, but can change on move/export)
    key = LCase$(NullToEmpty(mi.entryId))
    If Len(key) > 0 Then
        MessageIdentityKey = "E:" & key
        Exit Function
    End If

    ' 3. Fallback key (Time + Subject + Sender)
    key = Format$(EffectiveLocalTime(mi), "yyyy-mm-dd hh:nn:ss") & "|" & _
          LCase$(MakeSmartSubject(NullToEmpty(mi.subject))) & "|" & _
          LCase$(SafeGetSenderSmtp(mi))

    If Len(key) > 0 Then
        MessageIdentityKey = "F:" & key
    End If
End Function

Private Sub AddMailIfNew(ByRef out As Collection, ByVal mi As Outlook.MailItem, ByRef seen As Object)
    On Error Resume Next
    If mi Is Nothing Then Exit Sub
    If IsDraftMail(mi) Then Exit Sub    ' exclude drafts everywhere

    Dim key As String
    key = MessageIdentityKey(mi)
    If Len(key) = 0 Then Exit Sub

    If Not seen.Exists(key) Then
        out.Add mi
        seen(key) = True
    End If
End Sub

'=== SORTING ===================================================================

Private Function SortMailsByDateAsc(ByVal msgs As Collection) As Collection
    Set SortMailsByDateAsc = SortMailsByDate(msgs, True)
End Function

Private Function SortMailsByDateDesc(ByVal msgs As Collection) As Collection
    Set SortMailsByDateDesc = SortMailsByDate(msgs, False)
End Function

Private Function SortMailsByDate(ByVal msgs As Collection, ByVal ascending As Boolean) As Collection
    If msgs Is Nothing Or msgs.count <= 1 Then Set SortMailsByDate = msgs: Exit Function
    Dim n As Long: n = msgs.count

    Dim a() As Outlook.MailItem: ReDim a(1 To n)
    Dim t() As Date: ReDim t(1 To n)

    Dim i As Long
    For i = 1 To n
        Set a(i) = msgs(i)
        t(i) = EffectiveLocalTime(a(i))
    Next i

    QuickSortByTime a, t, 1, n, ascending

    Dim out As New Collection
    For i = 1 To n: out.Add a(i): Next i
    Set SortMailsByDate = out
End Function

Private Sub QuickSortByTime(ByRef a() As Outlook.MailItem, ByRef t() As Date, _
                            ByVal lo As Long, ByVal hi As Long, ByVal asc As Boolean)
    Dim i As Long, j As Long, p As Date
    Dim td As Date
    Dim o As Outlook.MailItem

    i = lo: j = hi
    p = t((lo + hi) \ 2)

    Do While i <= j
        If asc Then
            Do While t(i) < p: i = i + 1: Loop
            Do While t(j) > p: j = j - 1: Loop
        Else
            Do While t(i) > p: i = i + 1: Loop
            Do While t(j) < p: j = j - 1: Loop
        End If
        If i <= j Then
            td = t(i): t(i) = t(j): t(j) = td
            Set o = a(i): Set a(i) = a(j): Set a(j) = o
            i = i + 1: j = j - 1
        End If
    Loop

    If lo < j Then QuickSortByTime a, t, lo, j, asc
    If i < hi Then QuickSortByTime a, t, i, hi, asc
End Sub

Private Sub QuickSortStrings(ByRef a() As String, ByVal lo As Long, ByVal hi As Long)
    Dim i As Long, j As Long, p As String, tmp As String
    i = lo: j = hi: p = a((lo + hi) \ 2)
    Do While i <= j
        Do While a(i) < p: i = i + 1: Loop
        Do While a(j) > p: j = j - 1: Loop
        If i <= j Then
            tmp = a(i): a(i) = a(j): a(j) = tmp
            i = i + 1: j = j - 1
        End If
    Loop
    If lo < j Then QuickSortStrings a, lo, j
    If i < hi Then QuickSortStrings a, i, hi
End Sub

Private Function EffectiveLocalTime(ByVal mi As Outlook.MailItem) As Date
    On Error Resume Next
    Static cache As Object
    If cache Is Nothing Then
        Set cache = CreateObject("Scripting.Dictionary")
        cache.CompareMode = 1
    End If

    Dim key As String: key = LCase$(NullToEmpty(mi.entryId))
    If Len(key) = 0 Then key = CStr(mi.CreationTime) & "|" & LCase$(NullToEmpty(mi.subject))

    If cache.Exists(key) Then
        EffectiveLocalTime = cache(key)
        Exit Function
    End If

    Dim d As Date: d = mi.ReceivedTime
    If d = 0 Then d = mi.SentOn
    If d = 0 Then d = mi.CreationTime
    If d = 0 Then d = Now

    cache(key) = d
    EffectiveLocalTime = d
End Function

' Draft detector (used globally)
Private Function IsDraftMail(ByVal mi As Outlook.MailItem) As Boolean
    On Error Resume Next
    If mi Is Nothing Then Exit Function
    ' Unsent draft: not Sent, and no ReceivedTime/SentOn timestamps
    IsDraftMail = (Not mi.Sent) And mi.ReceivedTime = 0 And mi.SentOn = 0
End Function

'=== CONV_REF / NAMING ==========================================================

' Overflow-safe ShortHash (DJB2 variant)
Private Function ShortHash(ByVal s As String, ByVal n As Long) As String
    Dim h As Long: h = 5381
    Dim i As Long, ch As Long

    For i = 1 To Len(s)
        ch = AscW(mid$(s, i, 1)) And &HFFFF&

        ' DJB2: h = (h * 33) ^ ch
        ' Keep 25 bits before multiplying by 33 to stay within Long bounds
        h = ((h And &H1FFFFFF) * 33) Xor ch
        h = h And &H7FFFFFFF   ' mask back to positive 31-bit
    Next i

    Dim hexv As String
    hexv = hex$(h And &H7FFFFFFF)
    If Len(hexv) < n Then hexv = String$(n - Len(hexv), "0") & hexv
    ShortHash = Left$(hexv, n)
End Function

Private Function ConversationIncludesAddress(ByVal msgs As Collection, ByVal target As String) As Boolean
    Dim addr As String: addr = LCase$(Trim$(target))
    If Len(addr) = 0 Then
        ConversationIncludesAddress = False
        Exit Function
    End If
    Dim i As Long
    For i = 1 To msgs.count
        If ItemIncludesAddress(msgs(i), addr) Then ConversationIncludesAddress = True: Exit Function
    Next i
End Function

Private Function ComputeStableConvRef(ByVal initialMail As Outlook.MailItem, ByVal convMsgs As Collection) As String
    Dim hasFlag As Boolean
    hasFlag = ConversationIncludesAddress(convMsgs, FLAG_ADDR)
    Dim prefix As String: prefix = IIf(hasFlag, "TML-", "EML-")
    Dim anchor As Date: anchor = EffectiveLocalTime(initialMail)
    Dim key As String: key = GetConvKey(initialMail)
    Dim tag As String: tag = ShortHash(key, 6)
    ComputeStableConvRef = prefix & Format$(anchor, "yyyy-mm-dd") & "|" & tag
End Function

Private Function ItemIncludesAddress(ByVal mi As Outlook.MailItem, ByVal target As String) As Boolean
    Dim needle As String: needle = LCase$(Trim$(target))
    If Len(needle) = 0 Then Exit Function
    Dim hay As String
    hay = LCase$(SafeGetSenderSmtp(mi)) & "|" & _
          LCase$(JoinRecipientsOrHeader(mi, olTo)) & "|" & _
          LCase$(JoinRecipientsOrHeader(mi, olCC))
    ItemIncludesAddress = (InStr(1, hay, needle, vbTextCompare) > 0)
End Function

'=== WRITE CONVERSATION (atomic; UTF-8 w/ fallback) ============================

' RAG-oriented conversation writer (canonical Conversation.txt)
Private Function WriteConversationFileRag(ByVal pathTxt As String, ByVal sortedMsgs As Collection, _
                                          ByVal convRefRaw As String, ByVal smartSubj As String, _
                                          ByVal initialEntryID As String) As Boolean
    On Error GoTo EH

    ' Build content in memory
    Dim buf As String
    buf = "conv_ref: " & convRefRaw & " | subject: " & smartSubj & vbCrLf & vbCrLf

    ' Extra de-dup guard at file-write level using central identity key
    Dim seen As Object
    Set seen = CreateObject("Scripting.Dictionary")
    seen.CompareMode = 1

    Dim i As Long
    For i = 1 To sortedMsgs.count
        On Error GoTo SKIP_ONE

        Dim mi As Outlook.MailItem
        Set mi = sortedMsgs(i)
        If mi Is Nothing Then GoTo NEXT_I

        Dim k As String
        k = MessageIdentityKey(mi)
        If Len(k) = 0 Then GoTo NEXT_I
        If seen.Exists(k) Then GoTo NEXT_I
        seen(k) = True

        Dim isInitial As Boolean
        isInitial = (StrComp(NullToEmpty(mi.entryId), initialEntryID, vbBinaryCompare) = 0)

        buf = buf & String$(80, "-") & vbCrLf
        buf = buf & Format$(EffectiveLocalTime(mi), "yyyy-mm-dd hh:nn") & " | From: " & SafeGetSenderSmtp(mi)

        ' Keep original logic for To/Cc display
        If isInitial Then
            Dim sTo As String
            Dim sCc As String
            sTo = JoinRecipientsOrHeader(mi, olTo)
            sCc = JoinRecipientsOrHeader(mi, olCC)
            If Len(sTo) > 0 Then buf = buf & " | To: " & sTo
            If Len(sCc) > 0 Then buf = buf & " | Cc: " & sCc
        Else
            Dim onlyCc As String
            onlyCc = JoinRecipientsOrHeader(mi, olCC)
            If Len(onlyCc) > 0 Then buf = buf & " | Cc: " & onlyCc
        End If

        buf = buf & vbCrLf & vbCrLf

        Dim body As String
        ' Aggressive cleaning for RAG: signatures, disclaimers, banners, noise, whitespace
        body = CleanForEmbedding(ExtractTopMessageText(mi))
        If Len(Trim$(body)) = 0 Then body = "(no body)"

        buf = buf & body & vbCrLf & vbCrLf

        On Error GoTo 0
        GoTo NEXT_I

SKIP_ONE:
        Trace "  WARN: Skipped item during WriteConversationFileRag due to error (" & Err.Number & "): " & Err.Description
        Err.Clear
        On Error GoTo 0
NEXT_I:
    Next i

    ' Try UTF-8 via ADODB.Stream first (atomic .tmp + move)
    Dim fso As Object
    Set fso = CreateObject("Scripting.FileSystemObject")
    Dim tmpPath As String
    tmpPath = pathTxt & ".tmp"

    Dim ok As Boolean
    ok = SaveUtf8Atomic(buf, tmpPath, pathTxt)
    If Not ok Then
        ' Fallback: Unicode (UTF-16) via FSO, still atomic
        ok = SaveUnicodeAtomic(buf, tmpPath, pathTxt)
    End If

    WriteConversationFileRag = ok And fso.fileExists(pathTxt)
    Exit Function

EH:
    Trace "  ERROR in WriteConversationFileRag: (" & Err.Number & ") " & Err.Description
    WriteConversationFileRag = False
End Function

' Human-oriented conversation writer (Conversation_human.txt)
Private Function WriteConversationFileHuman(ByVal pathTxt As String, ByVal sortedMsgs As Collection, _
                                            ByVal convRefRaw As String, ByVal smartSubj As String, _
                                            ByVal initialEntryID As String) As Boolean
    On Error GoTo EH

    Dim buf As String
    buf = "conv_ref: " & convRefRaw & " | subject: " & smartSubj & vbCrLf & vbCrLf

    Dim seen As Object
    Set seen = CreateObject("Scripting.Dictionary")
    seen.CompareMode = 1

    Dim i As Long
    For i = 1 To sortedMsgs.count
        On Error GoTo SKIP_ONE

        Dim mi As Outlook.MailItem
        Set mi = sortedMsgs(i)
        If mi Is Nothing Then GoTo NEXT_I

        Dim k As String
        k = MessageIdentityKey(mi)
        If Len(k) = 0 Then GoTo NEXT_I
        If seen.Exists(k) Then GoTo NEXT_I
        seen(k) = True

        Dim isInitial As Boolean
        isInitial = (StrComp(NullToEmpty(mi.entryId), initialEntryID, vbBinaryCompare) = 0)

        buf = buf & String$(80, "-") & vbCrLf
        buf = buf & Format$(EffectiveLocalTime(mi), "yyyy-mm-dd hh:nn") & " | From: " & SafeGetSenderSmtp(mi)

        If isInitial Then
            Dim sTo As String
            Dim sCc As String
            sTo = JoinRecipientsOrHeader(mi, olTo)
            sCc = JoinRecipientsOrHeader(mi, olCC)
            If Len(sTo) > 0 Then buf = buf & " | To: " & sTo
            If Len(sCc) > 0 Then buf = buf & " | Cc: " & sCc
        Else
            Dim onlyCc As String
            onlyCc = JoinRecipientsOrHeader(mi, olCC)
            If Len(onlyCc) > 0 Then buf = buf & " | Cc: " & onlyCc
        End If

        buf = buf & vbCrLf & vbCrLf

        Dim body As String
        ' Gentler cleaning for humans: keep structure, but normalise punctuation and whitespace
        body = ExtractTopMessageText(mi)
        body = DecodeCommonHtmlEntities(body)
        body = NormalizeUnicodePunctuation(body)
        body = CollapseBlankLines(body)
        body = NormalizeWhitespace(body)

        If Len(Trim$(body)) = 0 Then body = "(no body)"

        buf = buf & body & vbCrLf & vbCrLf

        On Error GoTo 0
        GoTo NEXT_I

SKIP_ONE:
        Trace "  WARN: Skipped item during WriteConversationFileHuman due to error (" & Err.Number & "): " & Err.Description
        Err.Clear
        On Error GoTo 0
NEXT_I:
    Next i

    Dim fso As Object
    Set fso = CreateObject("Scripting.FileSystemObject")
    Dim tmpPath As String
    tmpPath = pathTxt & ".tmp"

    Dim ok As Boolean
    ok = SaveUtf8Atomic(buf, tmpPath, pathTxt)
    If Not ok Then
        ok = SaveUnicodeAtomic(buf, tmpPath, pathTxt)
    End If

    WriteConversationFileHuman = ok And fso.fileExists(pathTxt)
    Exit Function

EH:
    Trace "  ERROR in WriteConversationFileHuman: (" & Err.Number & ") " & Err.Description
    WriteConversationFileHuman = False
End Function

Private Function SaveUtf8Atomic(ByVal content As String, ByVal tmpPath As String, ByVal finalPath As String) As Boolean
    On Error GoTo EH

    Dim fso As Object
    Dim stmText As Object
    Dim stmOut As Object
    Dim byteCount As Long

    Set fso = CreateObject("Scripting.FileSystemObject")

    ' First write as UTF-8 text (this stream will have a BOM)
    Set stmText = CreateObject("ADODB.Stream")
    stmText.Type = 2                ' adTypeText
    stmText.Charset = "utf-8"
    stmText.Open
    stmText.WriteText content

    ' Switch to binary view so we can drop the BOM (EF BB BF) if present
    stmText.Position = 0
    stmText.Type = 1                ' adTypeBinary
    byteCount = stmText.Size

    Set stmOut = CreateObject("ADODB.Stream")
    stmOut.Type = 1                 ' adTypeBinary
    stmOut.Open

    If byteCount >= 3 Then
        ' Skip BOM bytes
        stmText.Position = 3
    Else
        stmText.Position = 0
    End If

    stmText.CopyTo stmOut
    stmText.Close

    stmOut.SaveToFile tmpPath, 2    ' adSaveCreateOverWrite
    stmOut.Close

    On Error Resume Next
    If fso.fileExists(finalPath) Then fso.DeleteFile finalPath, True
    fso.MoveFile tmpPath, finalPath
    If Err.Number <> 0 Then
        Err.Clear
        fso.CopyFile tmpPath, finalPath, True
        fso.DeleteFile tmpPath, True
    End If
    On Error GoTo 0

    SaveUtf8Atomic = fso.fileExists(finalPath)
    Exit Function

EH:
    On Error Resume Next
    If Not stmText Is Nothing Then
        If stmText.State = 1 Then stmText.Close
    End If
    If Not stmOut Is Nothing Then
        If stmOut.State = 1 Then stmOut.Close
    End If
    On Error GoTo 0
    SaveUtf8Atomic = False
End Function

Private Function SaveUnicodeAtomic(ByVal content As String, ByVal tmpPath As String, ByVal finalPath As String) As Boolean
    On Error GoTo EH
    Dim fso As Object: Set fso = CreateObject("Scripting.FileSystemObject")

    ' Write tmp as Unicode (UTF-16 LE)
    Dim ts As Object: Set ts = fso.CreateTextFile(tmpPath, True, True)
    ts.Write content
    ts.Close

    On Error Resume Next
    If fso.fileExists(finalPath) Then fso.DeleteFile finalPath, True
    fso.MoveFile tmpPath, finalPath
    If Err.Number <> 0 Then
        Err.Clear
        fso.CopyFile tmpPath, finalPath, True
        fso.DeleteFile tmpPath, True
    End If
    On Error GoTo 0

    SaveUnicodeAtomic = fso.fileExists(finalPath)
    Exit Function
EH:
    SaveUnicodeAtomic = False
End Function

Private Sub TryDeleteFolderIfEmpty(ByVal path As String)
    On Error Resume Next
    Dim fso As Object: Set fso = CreateObject("Scripting.FileSystemObject")
    If fso.FolderExists(path) Then
        Dim fld As Object: Set fld = fso.GetFolder(path)
        If fld.Files.count = 0 And fld.SubFolders.count = 0 Then fso.DeleteFolder path, True
    End If
End Sub

'=== BODY CLEANING ==============================================================

Private Function HtmlTopReplyToText(ByVal html As String) As String
    On Error Resume Next
    html = ToLF(html)

    Dim re As Object: Set re = CreateObject("VBScript.RegExp")
    re.Global = True: re.ignoreCase = True

    ' Strip script/style only
    re.pattern = "<script[\s\S]*?</script>|<style[\s\S]*?</style>"
    html = re.Replace(html, "")

    ' Map known history containers and <hr> to lightweight separators (don't delete the rest)
    ' Double-quoted attributes
    re.pattern = "<(div|blockquote)[^>]*(class|id)\s*=\s*""[^""]*(gmail_quote|gmail_attr|yahoo_quoted|outlookmessageheader)[^""]*""[^>]*>"
    html = re.Replace(html, "<br>----- quoted history -----<br>")
    ' Single-quoted attributes
    re.pattern = "<(div|blockquote)[^>]*(class|id)\s*=\s*'[^']*(gmail_quote|gmail_attr|yahoo_quoted|outlookmessageheader)[^']*'[^>]*>"
    html = re.Replace(html, "<br>----- quoted history -----<br>")

    re.pattern = "<hr[^>]*>"
    html = re.Replace(html, "<br>----- quoted history -----<br>")

    ' Now HTML -> text (preserve bullets and quote markers)
    HtmlTopReplyToText = HtmlToText(html)
End Function

' Preserve structure, bullets, and quoted lines ("> ")
Private Function HtmlToText(ByVal html As String) As String
    On Error Resume Next
    html = ToLF(html)

    Dim re As Object: Set re = CreateObject("VBScript.RegExp")
    re.Global = True: re.ignoreCase = True

    ' Remove script/style if still present
    re.pattern = "<script[\s\S]*?</script>|<style[\s\S]*?</style>"
    html = re.Replace(html, "")

    ' Line breaks
    re.pattern = "<\s*br\s*/?>": html = re.Replace(html, vbLf)

    ' Blockquotes -> mark quoted lines
    re.pattern = "<\s*blockquote[^>]*>": html = re.Replace(html, vbLf & "> ")
    re.pattern = "</\s*blockquote\s*>": html = re.Replace(html, vbLf)

    ' Close-tag boundaries -> newline
    re.pattern = "</\s*(p|div|li|tr|h[1-6]|table|blockquote)\s*>": html = re.Replace(html, vbLf)

    ' List items
    re.pattern = "<\s*li[^>]*>": html = re.Replace(html, vbLf & "- ")
    re.pattern = "</\s*li\s*>": html = re.Replace(html, vbLf)

    ' Strip the rest of tags
    re.pattern = "<[^>]+>": html = re.Replace(html, "")

    HtmlToText = NormalizeWhitespace(html)
End Function

' Safer extractor with signature-only fallback and hard empty guard
Private Function ExtractTopMessageText(ByVal mi As Outlook.MailItem) As String
    On Error Resume Next
    Dim html As String: html = NullToEmpty(mi.htmlBody)
    Dim textBody As String

    If Len(html) > 0 Then
        textBody = HtmlTopReplyToText(html)   ' non-destructive pre-pass
    Else
        textBody = NullToEmpty(mi.body)
    End If

    textBody = NormalizeWhitespace(textBody)

    ' Keep both variants and choose the richer one later if we fall back
    Dim preCut As String: preCut = textBody
    Dim topOnly As String

    topOnly = CutAtQuotedBlocks(preCut)
    topOnly = TrimSignature(topOnly)
    topOnly = TrimDisclaimers(topOnly)
    topOnly = NormalizeWhitespace(topOnly)

    ' If cutter produced an empty or signature-only stub, keep the pre-cut version (minus disclaimers)
    If LooksLikeSignatureOnly(topOnly) Then
        textBody = TrimDisclaimers(preCut)
    Else
        textBody = topOnly
    End If

    ExtractTopMessageText = Trim$(textBody)
End Function

' Keep “signature-only” stubs from replacing real content
Private Function LooksLikeSignatureOnly(ByVal s As String) As Boolean
    On Error Resume Next
    Dim t As String: t = Trim$(s)
    If Len(t) = 0 Then LooksLikeSignatureOnly = True: Exit Function
    Dim lines() As String: lines = Split(ToLF(t), vbLf)
    Dim score As Long: score = CountContactishLines(lines, LBound(lines), UBound(lines))
    LooksLikeSignatureOnly = (Len(t) < 80 And score >= 2)
End Function

Private Function CleanForEmbedding(ByVal s As String) As String
    s = DecodeCommonHtmlEntities(s)
    s = StripExternalMailBanners(s)    ' Multilingual: EN, AR, FR banners
    s = StripNoiseLines(s)             ' Drop orphan rule bullets (-, –, —, •, ·)
    s = StripContactFooterNoise(s)     ' Aggressive signature/footer removal (threshold=2)
    s = NormalizeUnicodePunctuation(s)
    s = CollapseBlankLines(s)
    s = NormalizeWhitespace(s)
    CleanForEmbedding = Trim$(s)
End Function

' Robust quote/history cutter with minimum-keep guard
Private Function CutAtQuotedBlocks(ByVal s As String) As String
    Dim lines() As String
    lines = Split(ToLF(s), vbLf)

    Dim i As Long, cutAt As Long: cutAt = -1
    Dim re As Object: Set re = CreateObject("VBScript.RegExp")
    re.Global = False: re.ignoreCase = True

    For i = LBound(lines) To UBound(lines)
        Dim L As String: L = Trim$(lines(i))
        If Len(L) = 0 Then GoTo cont

        ' Separators and common markers
        If L Like "-----Original Message-----*" Then cutAt = i - 1: Exit For
        If LCase$(L) = "----- quoted history -----" Then cutAt = i - 1: Exit For

        ' Forwarded block
        re.pattern = "^\s*-{2,}\s*forwarded message\s*-{2,}\s*$"
        If re.Test(L) Then cutAt = i - 1: Exit For
        If LCase$(Left$(L, 18)) = "forwarded message" Then cutAt = i - 1: Exit For

        ' "On … wrote:"
        re.pattern = "^\s*on .{1,160} wrote:\s*$"
        If re.Test(L) Then cutAt = i - 1: Exit For

        ' Outlook header cluster (can start with Sent:/From:/Date:)
        If IsHeaderStart(L) Then
            If CountHeaderishInRange(lines, i, MinLong(i + 10, UBound(lines))) >= 3 Then
                cutAt = i - 1: Exit For
            End If
        End If

        ' True quoted block lines (start-of-line >). Require at least 2 within next few lines
        If IsQuoteLine(L) Then
            If QuoteRunCount(lines, i, MinLong(i + 7, UBound(lines))) >= 2 Then
                cutAt = i - 1: Exit For
            End If
        End If
cont:
    Next i

    Dim kept As String
    If cutAt >= LBound(lines) Then
        kept = JoinUpTo(lines, cutAt)
    Else
        kept = JoinUpTo(lines, UBound(lines))
    End If

    ' Minimum keep guard
    If Len(Trim$(kept)) < 120 And cutAt >= 0 Then
        kept = KeepFirstMeaningfulChunk(lines, 12, 800)
    End If

    CutAtQuotedBlocks = kept
End Function

' Helpers for the cutter
Private Function IsHeaderStart(ByVal L As String) As Boolean
    Dim low As String: low = LCase$(L)
    IsHeaderStart = (Left$(low, 5) = "from:" Or _
                     Left$(low, 5) = "sent:" Or _
                     Left$(low, 3) = "to:" Or _
                     Left$(low, 3) = "cc:" Or _
                     Left$(low, 8) = "subject:" Or _
                     Left$(low, 5) = "date:")
End Function

Private Function CountHeaderishInRange(ByRef lines() As String, ByVal i1 As Long, ByVal i2 As Long) As Long
    Dim i As Long, c As Long, low As String
    For i = i1 To i2
        low = LCase$(Trim$(lines(i)))
        If Left$(low, 5) = "from:" Or Left$(low, 5) = "sent:" Or _
           Left$(low, 3) = "to:" Or Left$(low, 3) = "cc:" Or _
           Left$(low, 8) = "subject:" Or Left$(low, 5) = "date:" Then
            c = c + 1
        End If
    Next i
    CountHeaderishInRange = c
End Function

Private Function IsQuoteLine(ByVal L As String) As Boolean
    Dim re As Object: Set re = CreateObject("VBScript.RegExp")
    re.Global = False: re.ignoreCase = False
    re.pattern = "^\s*>+"
    IsQuoteLine = re.Test(L)
End Function

Private Function QuoteRunCount(ByRef lines() As String, ByVal i1 As Long, ByVal i2 As Long) As Long
    Dim i As Long, c As Long
    For i = i1 To i2
        If IsQuoteLine(lines(i)) Then c = c + 1
    Next i
    QuoteRunCount = c
End Function

' Keep the first meaningful paragraph (up to maxLines or maxChars), used as a guard.
Private Function KeepFirstMeaningfulChunk(ByRef lines() As String, ByVal maxLines As Long, ByVal maxChars As Long) As String
    Dim i As Long, total As Long, nonBlank As Long
    Dim stopAt As Long: stopAt = MinLong(UBound(lines), LBound(lines) + maxLines - 1)
    For i = LBound(lines) To stopAt
        If Trim$(lines(i)) <> "" Then nonBlank = nonBlank + 1
        total = total + Len(lines(i)) + 2
        If Trim$(lines(i)) = "" And nonBlank >= 2 Then Exit For
        If total >= maxChars Then Exit For
    Next i
    Dim lastIdx As Long: lastIdx = i - 1
    If lastIdx < LBound(lines) Then lastIdx = LBound(lines)
    If lastIdx > stopAt Then lastIdx = stopAt
    KeepFirstMeaningfulChunk = JoinUpTo(lines, lastIdx)
End Function

' Trim at signature separators and common closings (bottom-up)
Private Function TrimSignature(ByVal s As String) As String
    On Error Resume Next
    If Len(s) = 0 Then TrimSignature = s: Exit Function
    s = ToLF(s)
    Dim lines() As String: lines = Split(s, vbLf)
    Dim re As Object: Set re = CreateObject("VBScript.RegExp")
    re.ignoreCase = True: re.Global = False
    Dim i As Long, cutAt As Long: cutAt = -1
    For i = LBound(lines) To UBound(lines)
        Dim L As String: L = Trim$(lines(i))
        re.pattern = "^\s*(--|__|\*{2,}|-{2,}|_{2,})\s*$": If re.Test(L) Then cutAt = i - 1: Exit For
        re.pattern = "^\s*(sent\s+from\s+my\s+(iphone|ipad|android|mobile)|get\s+outlook\s+for\s+(ios|android))\s*$": If re.Test(L) Then cutAt = i - 1: Exit For
        If LCase$(Left$(L, 12)) = "best regards" Or LCase$(Left$(L, 12)) = "kind regards" Or _
           LCase$(Left$(L, 7)) = "regards" Or LCase$(Left$(L, 6)) = "thanks" Or _
           LCase$(Left$(L, 9)) = "thank you" Or LCase$(Left$(L, 9)) = "sincerely" Or _
           LCase$(Left$(L, 6)) = "cheers" Then
            Dim maxLook As Long, score As Long: maxLook = MinLong(i + 12, UBound(lines))
            score = CountContactishLines(lines, i + 1, maxLook)
            If score >= 2 Then cutAt = i - 1: Exit For
        End If
    Next i
    If cutAt >= 0 Then TrimSignature = Trim$(JoinUpTo(lines, cutAt)) Else TrimSignature = Replace$(s, vbLf, vbCrLf)
End Function

' Tail-only disclaimer trimming (no mid-body amputations)
Private Function TrimDisclaimers(ByVal s As String) As String
    On Error Resume Next
    If Len(s) = 0 Then
        TrimDisclaimers = s
        Exit Function
    End If

    Dim t As String
    t = ToLF(s)

    Dim lines() As String
    lines = Split(t, vbLf)

    Dim i As Long

    Dim re As Object
    Set re = CreateObject("VBScript.RegExp")
    re.Global = False
    re.ignoreCase = True

    ' English + Arabic disclaimer markers (????)
    re.pattern = "(confidentiality notice|" & _
                 "if you are not the intended recipient|" & _
                 "this (e-?mail|message).{0,300}confidential|" & _
                 "\bdisclaimer\b|" & _
                 ChrW$(&H62A) & ChrW$(&H646) & ChrW$(&H635) & ChrW$(&H644) & ")" ' ????

    For i = UBound(lines) To LBound(lines) Step -1
        Dim L As String
        L = Trim$(lines(i))
        If Len(L) = 0 Then GoTo cont

        If re.Test(L) Then
            ' Only treat as disclaimer if there is a block of legalese below (at least 3 non-blank lines)
            Dim j As Long, nonBlank As Long
            nonBlank = 0
            For j = i To MinLong(i + 40, UBound(lines))
                If Trim$(lines(j)) <> "" Then nonBlank = nonBlank + 1
            Next j

            If nonBlank >= 3 Then
                TrimDisclaimers = Trim$(JoinUpTo(lines, i - 1))
                Exit Function
            End If
        End If
cont:
    Next i

    TrimDisclaimers = s
End Function

' External banner / warning stripper (English + Arabic + multilingual)
Private Function StripExternalMailBanners(ByVal s As String) As String
    Dim re As Object
    Set re = CreateObject("VBScript.RegExp")
    re.Global = True
    re.ignoreCase = True
    re.multiLine = True

    ' === ENGLISH PATTERNS (with optional trailing newline consumption) ===
    re.pattern = "^\s*CAUTION:.*$(\r\n|\n)?"
    s = re.Replace(s, "")

    re.pattern = "^\s*this\s+email\s+originated\s+from\s+outside.*$(\r\n|\n)?"
    s = re.Replace(s, "")

    re.pattern = "^\s*external\s+email\s+warning:.*$(\r\n|\n)?"
    s = re.Replace(s, "")

    re.pattern = "^\s*PUBLIC\s*$(\r\n|\n)?"
    s = re.Replace(s, "")

    re.pattern = "^\s*this\s+email\s+was\s+sent\s+from\s+outside.*$(\r\n|\n)?"
    s = re.Replace(s, "")

    re.pattern = "^\s*this\s+message\s+originated\s+from\s+outside.*$(\r\n|\n)?"
    s = re.Replace(s, "")

    re.pattern = "^\s*warning:\s*this\s+(email|message)\s+(is|was|has).*outside.*$(\r\n|\n)?"
    s = re.Replace(s, "")

    re.pattern = "^\s*do\s+not\s+click\s+(on\s+)?links\s+or\s+open\s+attachments.*$(\r\n|\n)?"
    s = re.Replace(s, "")

    re.pattern = "^\s*please\s+consider\s+the\s+environment\s+before\s+printing\s+this\s+(e-?mail|email).*$(\r\n|\n)?"
    s = re.Replace(s, "")

    ' === ARABIC PATTERNS (Chalhoub Group and common variants) ===
    ' "This email was sent from outside the Chalhoub Group"
    re.pattern = "^\s*" & ChrW$(&H62A) & ChrW$(&H645) & " " & ChrW$(&H625) & ChrW$(&H631) & ChrW$(&H633) & ChrW$(&H627) & ChrW$(&H644) & " " & ChrW$(&H647) & ChrW$(&H630) & ChrW$(&H627) & " " & ChrW$(&H627) & ChrW$(&H644) & ChrW$(&H628) & ChrW$(&H631) & ChrW$(&H64A) & ChrW$(&H62F) & " " & ChrW$(&H627) & ChrW$(&H644) & ChrW$(&H625) & ChrW$(&H644) & ChrW$(&H643) & ChrW$(&H62A) & ChrW$(&H631) & ChrW$(&H648) & ChrW$(&H646) & ChrW$(&H64A) & " " & ChrW$(&H645) & ChrW$(&H646) & " " & ChrW$(&H62E) & ChrW$(&H627) & ChrW$(&H631) & ChrW$(&H62C) & ".*$(\r\n|\n)?"
    s = re.Replace(s, "")

    ' "Do not click on links or open attachments" (Arabic)
    re.pattern = "^\s*" & ChrW$(&H644) & ChrW$(&H627) & " " & ChrW$(&H62A) & ChrW$(&H646) & ChrW$(&H642) & ChrW$(&H631) & " " & ChrW$(&H639) & ChrW$(&H644) & ChrW$(&H649) & " " & ChrW$(&H627) & ChrW$(&H644) & ChrW$(&H631) & ChrW$(&H648) & ChrW$(&H627) & ChrW$(&H628) & ChrW$(&H637) & " " & ChrW$(&H623) & ChrW$(&H648) & " " & ChrW$(&H62A) & ChrW$(&H641) & ChrW$(&H62A) & ChrW$(&H62D) & " " & ChrW$(&H627) & ChrW$(&H644) & ChrW$(&H645) & ChrW$(&H631) & ChrW$(&H641) & ChrW$(&H642) & ChrW$(&H627) & ChrW$(&H62A) & ".*$(\r\n|\n)?"
    s = re.Replace(s, "")

    ' Fallback: Simpler patterns using key Arabic words
    ' Match lines starting with Arabic word for "sent" (??) followed by email-related text
    re.pattern = "^\s*" & ChrW$(&H62A) & ChrW$(&H645) & ".*" & ChrW$(&H625) & ChrW$(&H644) & ChrW$(&H643) & ChrW$(&H62A) & ChrW$(&H631) & ChrW$(&H648) & ChrW$(&H646) & ChrW$(&H64A) & ".*" & ChrW$(&H62E) & ChrW$(&H627) & ChrW$(&H631) & ChrW$(&H62C) & ".*$(\r\n|\n)?"
    s = re.Replace(s, "")

    ' Match lines containing "?? ????" (do not click)
    re.pattern = "^\s*" & ChrW$(&H644) & ChrW$(&H627) & " " & ChrW$(&H62A) & ChrW$(&H646) & ChrW$(&H642) & ChrW$(&H631) & ".*$(\r\n|\n)?"
    s = re.Replace(s, "")

    ' === FRENCH PATTERNS ===
    re.pattern = "^\s*attention\s*[:-].*provient\s+de\s+l'ext" & ChrW$(&HE9) & "rieur.*$(\r\n|\n)?"
    s = re.Replace(s, "")

    re.pattern = "^\s*ce\s+(courriel|message|e-?mail)\s+(a\s+)?" & ChrW$(&HE9) & "t" & ChrW$(&HE9) & "\s+envoy" & ChrW$(&HE9) & ".*ext" & ChrW$(&HE9) & "rieur.*$(\r\n|\n)?"
    s = re.Replace(s, "")

    StripExternalMailBanners = s
End Function

' Drop one-character rule/bullet noise lines that slip through
Private Function StripNoiseLines(ByVal s As String) As String
    Dim a() As String: a = Split(ToLF(s), vbLf)
    Dim i As Long
    Dim kept As New Collection
    For i = LBound(a) To UBound(a)
        Dim L As String: L = Trim$(a(i))
        If L = "-" Or L = "–" Or L = "—" Or L = "•" Or L = "·" Then
            ' skip
        Else
            kept.Add a(i)
        End If
    Next i
    If kept.count = 0 Then
        StripNoiseLines = ""
        Exit Function
    End If
    Dim out() As String: ReDim out(0 To kept.count - 1)
    For i = 1 To kept.count
        out(i - 1) = kept(i)
    Next i
    StripNoiseLines = Replace$(Join(out, vbCrLf), vbLf, vbCrLf)
End Function

Private Function StripContactFooterNoise(ByVal s As String) As String
    Dim lines() As String: lines = Split(ToLF(s), vbLf)
    Dim i As Long, cutAt As Long: cutAt = -1
    For i = UBound(lines) To LBound(lines) Step -1
        Dim L As String: L = Trim$(lines(i))
        Dim low As String: low = LCase$(L)
        ' Aggressive signature trimming for common closings (threshold=2 for RAG optimization)
        ' Matches: best regards, kind regards, regards, thanks, thank you, sincerely, best, cheers, warm regards
        If low Like "best regards*" Or low Like "kind regards*" Or _
           low Like "warm regards*" Or low Like "regards*" Or _
           low Like "thanks*" Or low Like "thank you*" Or _
           low Like "sincerely*" Or low Like "cheers*" Or _
           low Like "best,*" Or low = "best" Then
            Dim score As Long: score = CountContactishLines(lines, i + 1, MinLong(i + 12, UBound(lines)))
            ' Lowered threshold from 3 to 2 for more aggressive RAG cleaning
            If score >= 2 Then cutAt = i - 1: Exit For
        End If
    Next i
    If cutAt >= 0 Then StripContactFooterNoise = JoinUpTo(lines, cutAt) Else StripContactFooterNoise = s
End Function

Private Function CountContactishLines(ByRef lines() As String, ByVal i1 As Long, ByVal i2 As Long) As Long
    Dim re As Object: Set re = CreateObject("VBScript.RegExp")
    re.ignoreCase = True: re.Global = False
    Dim i As Long, c As Long, L As String
    For i = i1 To i2
        L = Trim$(lines(i)): If Len(L) = 0 Then GoTo nxt
        If InStr(LCase$(L), "@") > 0 Then c = c + 1: GoTo nxt
        re.pattern = "(tel|mobile|mob|phone|p\.?o\.?\s*box|www\.|https?://)": If re.Test(L) Then c = c + 1: GoTo nxt
        re.pattern = "\b([a-z0-9\-]+\.)+[a-z]{2,}\b": If re.Test(L) Then c = c + 1: GoTo nxt
        re.pattern = "^\+?\d[\d\s\-\(\)]{5,}$": If re.Test(L) Then c = c + 1: GoTo nxt
nxt:
    Next i
    CountContactishLines = c
End Function

Private Function DecodeCommonHtmlEntities(ByVal s As String) As String
    ' Common substitutions
    s = Replace$(s, "&nbsp;", " ")
    s = Replace$(s, "&#160;", " ")
    s = Replace$(s, "&amp;", "&")
    s = Replace$(s, "&lt;", "<")
    s = Replace$(s, "&gt;", ">")
    s = Replace$(s, "&quot;", """")
    s = Replace$(s, "&apos;", "'")

    ' Unicode space normalization
    s = Replace$(s, ChrW(&HA0), " ")
    s = Replace$(s, ChrW(&H200B), "")
    s = Replace$(s, ChrW(&HFEFF), "")

    Dim re As Object
    Set re = CreateObject("VBScript.RegExp")
    re.Global = True
    re.ignoreCase = True

    Dim m As Object, coll As Object
    Dim v As Long

    ' decimal entities (guarded against overflow)
    re.pattern = "&#(\d+);"
    Set coll = re.Execute(s)

    Dim decVal As String
    For Each m In coll
        decVal = m.SubMatches(0)
        ' Guard 1: Length check. Max VBA Long is 2147483647 (10 digits).
        If Len(decVal) > 0 And Len(decVal) <= 10 Then
            If IsNumeric(decVal) Then
                On Error Resume Next
                v = CLng(decVal)
                If Err.Number = 0 Then
                    ' Guard 2: Range check (valid Unicode BMP characters)
                    If v >= 0 And v <= &HFFFF& Then
                        s = Replace$(s, m.value, ChrW(v))
                    End If
                Else
                    Err.Clear
                End If
                On Error GoTo 0
            End If
        End If
    Next m

    ' hex entities (guarded against overflow)
    re.pattern = "&#x([0-9A-Fa-f]+);"
    Set coll = re.Execute(s)

    Dim hexVal As String
    For Each m In coll
        hexVal = m.SubMatches(0)
        ' Guard 1: Length check. Max Hex for Long is 7FFFFFFF (8 chars).
        If Len(hexVal) > 0 And Len(hexVal) <= 8 Then
            On Error Resume Next
            v = CLng("&H" & hexVal)
            If Err.Number = 0 Then
                If v >= 0 And v <= &HFFFF& Then
                    s = Replace$(s, m.value, ChrW(v))
                End If
            Else
                Err.Clear
            End If
            On Error GoTo 0
        End If
    Next m

    DecodeCommonHtmlEntities = s
End Function

' Added non-breaking hyphen and soft hyphen normalization
Private Function NormalizeUnicodePunctuation(ByVal s As String) As String
    s = Replace$(s, ChrW(&H2018), "'"): s = Replace$(s, ChrW(&H2019), "'")
    s = Replace$(s, ChrW(&H201C), """"): s = Replace$(s, ChrW(&H201D), """")
    s = Replace$(s, ChrW(&H2013), "-"): s = Replace$(s, ChrW(&H2014), "-"): s = Replace$(s, ChrW(&H2026), "...")
    s = Replace$(s, ChrW(&H2011), "-")  ' non-breaking hyphen ? hyphen
    s = Replace$(s, ChrW(&HAD), "")     ' soft hyphen ? drop
    s = Replace$(s, ChrW(&H202F), " ")  ' narrow no-break space ? space
    NormalizeUnicodePunctuation = s
End Function

Private Function CollapseBlankLines(ByVal s As String) As String
    Dim re As Object: Set re = CreateObject("VBScript.RegExp")
    re.ignoreCase = True: re.Global = True
    re.pattern = "(\r\n|\n){4,}"
    CollapseBlankLines = re.Replace(s, vbCrLf & vbCrLf)
End Function

' Linear-time whitespace normalization
Private Function NormalizeWhitespace(ByVal s As String) As String
    If Len(s) = 0 Then NormalizeWhitespace = s: Exit Function
    s = ToLF(s)
    s = Replace$(s, ChrW(&HA0), " ")
    s = Replace$(s, ChrW(&H200B), "")
    s = Replace$(s, ChrW(&HFEFF), "")
    s = Replace$(s, vbTab, " ")

    Dim re As Object: Set re = CreateObject("VBScript.RegExp")
    re.Global = True: re.multiLine = True: re.ignoreCase = True

    ' Trim trailing spaces per line
    re.pattern = "[ \t]+$"
    s = re.Replace(s, "")

    ' Collapse 2+ spaces to one
    re.pattern = " {2,}"
    s = re.Replace(s, " ")

    ' Remove space before newline
    s = Replace$(s, " " & vbLf, vbLf)

    ' Normalize newline convention and final trim
    s = Replace$(s, vbLf, vbCrLf)
    NormalizeWhitespace = Trim$(s)
End Function

'=== RECIPIENTS =================================================================

Private Function JoinRecipients(ByVal recips As Outlook.recipients, ByVal typ As OlMailRecipientType) As String
    On Error Resume Next
    If recips Is Nothing Then Exit Function
    Dim i As Long, buf As String, part As String
    For i = 1 To recips.count
        If recips(i).Type = typ Then
            part = NormalizeSmtpAddress(recips(i).AddressEntry)
            If Len(part) = 0 Then part = LCase$(NullToEmpty(recips(i).Address))
            If Len(part) > 0 Then
                If Len(buf) > 0 Then buf = buf & "; "
                buf = buf & Trim$(part)
            End If
        End If
    Next i
    JoinRecipients = buf
End Function

Private Function JoinRecipientsOrHeader(ByVal mi As Outlook.MailItem, ByVal typ As OlMailRecipientType) As String
    On Error Resume Next
    Dim s As String: s = JoinRecipients(mi.recipients, typ)
    If Len(s) = 0 Then
        If typ = olTo Then s = NullToEmpty(mi.To)
        If typ = olCC Then s = NullToEmpty(mi.cc)
    End If
    JoinRecipientsOrHeader = s
End Function

Private Function SafeGetSenderSmtp(ByVal mi As Outlook.MailItem) As String
    On Error Resume Next
    If mi Is Nothing Then Exit Function
    Dim ae As Object: Set ae = mi.sender
    If Not ae Is Nothing Then SafeGetSenderSmtp = NormalizeSmtpAddress(ae)
    If Len(SafeGetSenderSmtp) = 0 Then SafeGetSenderSmtp = LCase$(NullToEmpty(mi.SenderEmailAddress))
    If Len(SafeGetSenderSmtp) = 0 Then SafeGetSenderSmtp = "(unknown)"
End Function

Private Function NormalizeSmtpAddress(ByVal ae As Object) As String
    On Error Resume Next
    Dim smtp As String
    If ae Is Nothing Then Exit Function
    If TypeName(ae) = "AddressEntry" Then
        If LCase$(NullToEmpty(ae.Type)) = "ex" Then
            Dim exUser As Object: Set exUser = ae.GetExchangeUser
            If Not exUser Is Nothing Then smtp = NullToEmpty(exUser.PrimarySmtpAddress)
            If Len(smtp) = 0 Then
                Dim exDL As Object: Set exDL = ae.GetExchangeDistributionList
                If Not exDL Is Nothing Then smtp = NullToEmpty(exDL.PrimarySmtpAddress)
            End If
            If Len(smtp) = 0 Then
                Dim pa As Object: Set pa = ae.PropertyAccessor
                smtp = NullToEmpty(pa.GetProperty("http://schemas.microsoft.com/mapi/proptag/0x39FE001E"))
            End If
        Else
            smtp = NullToEmpty(ae.Address)
        End If
    ElseIf TypeName(ae) = "Recipient" Then
        smtp = NormalizeSmtpAddress(ae.AddressEntry)
    ElseIf TypeName(ae) = "MailItem" Then
        smtp = NormalizeSmtpAddress(ae.sender)
    End If
    If Len(smtp) = 0 Then Exit Function
    NormalizeSmtpAddress = LCase$(StripAngleBrackets(smtp))
End Function

Private Function StripAngleBrackets(ByVal s As String) As String
    Dim p1 As Long, p2 As Long
    p1 = InStr(1, s, "<"): p2 = InStr(1, s, ">")
    If p1 > 0 And p2 > p1 Then StripAngleBrackets = mid$(s, p1 + 1, p2 - p1 - 1) Else StripAngleBrackets = s
End Function

'=== PATH / UTILITIES ===========================================================

Private Function ResolveOutRoot(ByVal Preferred As String) As String
    On Error Resume Next
    Dim fso As Object: Set fso = CreateObject("Scripting.FileSystemObject")
    If Len(Preferred) > 0 Then
        If fso.FolderExists(Preferred) Then
            ResolveOutRoot = Preferred: Exit Function
        End If
        ' ensure deep creation instead of single CreateFolder
        ResolveOutRoot = EnsureFolderDeep(Preferred)
        If fso.FolderExists(ResolveOutRoot) Then Exit Function
    End If
    Dim fb As String: fb = Environ$("LOCALAPPDATA")
    If Len(fb) = 0 Then fb = Environ$("TEMP")
    If Len(fb) = 0 Then fb = "C:\Temp"
    ResolveOutRoot = EnsureFolderDeep(AppendPath(fb, "ConvTxt_Builder"))
End Function

' Run-level out-root with folder picker (shown once per run)
Private Function GetRunOutRoot() As String
    Static cached As String
    If Len(cached) > 0 Then
        GetRunOutRoot = cached
        Exit Function
    End If

    Dim basePref As String
    basePref = OUT_ROOT

    Dim chosen As String
    chosen = PickOutRootFolder(basePref)

    If Len(chosen) = 0 Then
        cached = ResolveOutRoot(basePref)
    Else
        cached = ResolveOutRoot(chosen)
    End If

    GetRunOutRoot = cached
End Function

' Folder picker using Office FileDialog (FolderPicker), with safe fallback
Private Function PickOutRootFolder(ByVal defaultPath As String) As String
    On Error GoTo EH
    Dim app As Outlook.Application
    Set app = Application

    Dim fd As Object
    Set fd = app.FileDialog(4) ' 4 = msoFileDialogFolderPicker

    With fd
        .title = "Select root folder for Conversation export"
        .AllowMultiSelect = False

        Dim init As String
        init = Trim$(defaultPath)
        If Len(init) = 0 Then
            init = Environ$("USERPROFILE")
            If Len(init) > 0 Then
                init = init & "\Desktop"
            End If
        End If
        If Len(init) > 0 Then
            .InitialFileName = init
        End If

        If .Show = -1 Then
            PickOutRootFolder = .SelectedItems(1)
        Else
            PickOutRootFolder = ""
        End If
    End With

    Exit Function
EH:
    ' any error ? no choice; caller falls back
    PickOutRootFolder = ""
End Function

Private Function AppendPath(ByVal base As String, ByVal leaf As String) As String
    If Right$(base, 1) = "\" Then AppendPath = base & leaf Else AppendPath = base & "\" & leaf
End Function

Private Function EnsureFolderDeep(ByVal path As String) As String
    Dim fso As Object: Set fso = CreateObject("Scripting.FileSystemObject")
    If fso.FolderExists(path) Then EnsureFolderDeep = path: Exit Function
    Dim parts() As String: parts = Split(path, "\")
    Dim cur As String: cur = parts(0)
    Dim i As Long
    For i = 1 To UBound(parts)
        cur = cur & "\" & parts(i)
        If Not fso.FolderExists(cur) Then On Error Resume Next: fso.CreateFolder cur: On Error GoTo 0
    Next i
    EnsureFolderDeep = path
End Function

' Format a Date as ISO-8601 "YYYY-MM-DDTHH:MM:SSZ".
' We keep the EffectiveLocalTime value but serialize it in canonical shape.
Private Function FormatIsoUtcString(ByVal d As Date) As String
    On Error Resume Next

    If d = 0 Then
        FormatIsoUtcString = ""
        Exit Function
    End If

    FormatIsoUtcString = Format$(d, "yyyy-mm-dd\Thh:nn:ss") & "Z"
End Function


' Return just the leaf folder name (relative folder name for manifest.folder)
Private Function GetFolderLeafName(ByVal path As String) As String
    On Error Resume Next
    Dim fso As Object
    Set fso = CreateObject("Scripting.FileSystemObject")
    GetFolderLeafName = fso.GetFileName(path)
End Function

Private Function SanitizeName(ByVal s As String) As String
    If Len(Trim$(s)) = 0 Then s = "unnamed"
    Dim bad As Variant: bad = Array("<", ">", ":", """", "/", "\", "|", "?", "*")
    Dim i As Long: For i = LBound(bad) To UBound(bad): s = Replace$(s, bad(i), IIf(bad(i) = ":", "·", "_")): Next i
    Do While Len(s) > 0 And (Right$(s, 1) = "." Or Right$(s, 1) = " "): s = Left$(s, Len(s) - 1): Loop
    Dim reserved As Variant: reserved = Array("CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9")
    Dim up As String: up = UCase$(s)
    For i = LBound(reserved) To UBound(reserved)
        If up = reserved(i) Or up Like reserved(i) & ".*" Or up Like reserved(i) & " *" Then s = "_" & s: Exit For
    Next i
    SanitizeName = s
End Function

' Old naive budget helper (kept for folder naming), still used—but see attachment-safe variant below.
Private Function FitForPathBudget(ByVal component As String, ByVal basePath As String, ByVal extraSuffix As String) As String
    Dim budget As Long: budget = MAX_PATH_BUDGET
    Dim allow As Long: allow = budget - Len(basePath) - 1 - Len(extraSuffix)
    If allow < 1 Then allow = 1
    If Len(component) > allow Then FitForPathBudget = Left$(component, allow) Else FitForPathBudget = component
End Function

' budget helper for FILES that PRESERVES EXTENSION and adds a short hash when truncated
Private Function FitFileNameForBudgetPreserveExt(ByVal fileName As String, ByVal basePath As String, Optional ByVal hashLen As Long = 6) As String
    Dim nm As String: nm = SanitizeName(fileName)
    Dim allow As Long
    allow = MAX_PATH_BUDGET - Len(basePath) - 1 ' "\" between basePath and filename
    If allow < 1 Then allow = 1

    If Len(nm) <= allow Then
        FitFileNameForBudgetPreserveExt = nm
        Exit Function
    End If

    ' Split stem/ext
    Dim dotPos As Long: dotPos = InStrRev(nm, ".")
    Dim stem As String, ext As String
    If dotPos > 0 Then
        stem = Left$(nm, dotPos - 1)
        ext = mid$(nm, dotPos) ' includes dot
    Else
        stem = nm
        ext = ""
    End If

    Dim tag As String: tag = "~" & ShortHash(nm, hashLen)

    Dim need As Long: need = Len(ext) + Len(tag) + 1  ' +1 at least for one stem char
    If allow <= Len(ext) Then
        ' Edge case: almost no headroom; keep tail of extension only
        FitFileNameForBudgetPreserveExt = Right$(ext, allow)
        Exit Function
    End If

    Dim bud As Long: bud = allow - Len(ext) - Len(tag)
    If bud < 1 Then
        ' Not enough room for hash tag; drop it
        bud = allow - Len(ext)
        If bud < 1 Then
            FitFileNameForBudgetPreserveExt = Right$(ext, allow)
            Exit Function
        End If
        FitFileNameForBudgetPreserveExt = Left$(stem, bud) & ext
        Exit Function
    End If

    FitFileNameForBudgetPreserveExt = Left$(stem, bud) & tag & ext
End Function

Private Function MakeSmartSubject(ByVal s As String) As String
    Dim t As String: t = Trim$(s)
    Dim prefixes As Variant: prefixes = Array("re: ", "fw: ", "fwd: ", "sv: ", "aw: ", "wg: ", "antw: ", "tr: ", "rv: ")
    Dim changed As Boolean
    Do
        changed = False
        Dim i As Long
        For i = LBound(prefixes) To UBound(prefixes)
            If LCase$(Left$(t, Len(prefixes(i)))) = prefixes(i) Then t = mid$(t, Len(prefixes(i)) + 1): changed = True: Exit For
        Next i
    Loop While changed
    If Len(t) = 0 Then MakeSmartSubject = "_" Else MakeSmartSubject = t
End Function

Private Function NullToEmpty(ByVal v As Variant) As String
    If IsNull(v) Or IsEmpty(v) Then NullToEmpty = "" Else NullToEmpty = CStr(v)
End Function

Private Function MinLong(ByVal a As Long, ByVal b As Long) As Long
    If a < b Then MinLong = a Else MinLong = b
End Function

' Normalize CRLF/CR ? LF once
Private Function ToLF(ByVal s As String) As String
    If Len(s) = 0 Then Exit Function
    s = Replace$(s, vbCrLf, vbLf)
    s = Replace$(s, vbCr, vbLf)
    ToLF = s
End Function

' Small helper to reduce RegExp boilerplate (optional)
Private Function NewRegex(Optional ByVal ignoreCase As Boolean = True, _
                          Optional ByVal globalMatch As Boolean = True, _
                          Optional ByVal multiLine As Boolean = False) As Object
    Dim re As Object
    Set re = CreateObject("VBScript.RegExp")
    re.ignoreCase = ignoreCase
    re.Global = globalMatch
    re.multiLine = multiLine
    Set NewRegex = re
End Function

' Fast join of a slice of lines
Private Function JoinUpTo(ByRef lines() As String, ByVal lastIdx As Long) As String
    Dim firstIdx As Long: firstIdx = LBound(lines)
    If lastIdx < firstIdx Then
        JoinUpTo = ""
        Exit Function
    End If
    If lastIdx > UBound(lines) Then lastIdx = UBound(lines)
    Dim n As Long: n = lastIdx - firstIdx + 1
    Dim buf() As String: ReDim buf(0 To n - 1)
    Dim i As Long
    For i = 0 To n - 1
        buf(i) = lines(firstIdx + i)
    Next i
    JoinUpTo = Join(buf, vbCrLf)
End Function

'===============================================================================
'=== CSV LOGGING ================================================================
'===============================================================================

Private Function EnsureCsvLog(ByVal attDir As String) As String
    Dim fso As Object: Set fso = CreateObject("Scripting.FileSystemObject")
    Dim p As String: p = AppendPath(attDir, "attachments_log.csv")
    If Not fso.fileExists(p) Then
        AppendCsvRow p, Array( _
            "run_date", "run_time", "action", "filename", "size_bytes", "attachment_kind", _
            "sender", "mail_subject", "mail_time", "attachment_mod_time", "mail_entryid", "saved_path" _
        )
    End If
    EnsureCsvLog = p
End Function

Private Function LoadCsvManifest(ByVal logPath As String) As Object
    On Error Resume Next
    Dim fso As Object: Set fso = CreateObject("Scripting.FileSystemObject")
    Dim dict As Object: Set dict = CreateObject("Scripting.Dictionary")
    dict.CompareMode = 1
    If Not fso.fileExists(logPath) Then Set LoadCsvManifest = dict: Exit Function

    Dim f As Integer: f = FreeFile
    Open logPath For Input As #f
    If Err.Number <> 0 Then Set LoadCsvManifest = dict: Exit Function
    Dim header As String
    If Not EOF(f) Then Line Input #f, header

    Dim h() As String: h = CsvSplitToArray(header)
    Dim iAction As Long: iAction = CsvIndexOf(h, "action")
    Dim iFilename As Long: iFilename = CsvIndexOf(h, "filename")
    Dim iAttachMod As Long: iAttachMod = CsvIndexOf(h, "attachment_mod_time")
    If iAction < 0 Or iFilename < 0 Or iAttachMod < 0 Then Close #f: Set LoadCsvManifest = dict: Exit Function

    Dim line As String, fields() As String
    Do While Not EOF(f)
        Line Input #f, line
        If Len(line) = 0 Then GoTo cont
        fields = CsvSplitToArray(line)
        If UBound(fields) >= iAttachMod Then
            Dim action As String: action = LCase$(fields(iAction))
            If action = "saved_new" Or action = "overwritten_newer" Then
                Dim fn As String: fn = LCase$(fields(iFilename))
                Dim ts As String: ts = fields(iAttachMod)
                If Len(fn) > 0 And Len(ts) > 0 Then dict(fn) = ts
            End If
        End If
cont:
    Loop
    Close #f
    Set LoadCsvManifest = dict
End Function

Private Function CsvIndexOf(ByRef arr() As String, ByVal name As String) As Long
    Dim i As Long
    For i = LBound(arr) To UBound(arr)
        If LCase$(arr(i)) = LCase$(name) Then CsvIndexOf = i: Exit Function
    Next i
    CsvIndexOf = -1
End Function

Private Sub AppendCsvRow(ByVal pathCsv As String, ByVal fields As Variant)
    On Error Resume Next
    Dim f As Integer: f = FreeFile
    Open pathCsv For Append As #f
    Print #f, Join(CsvEscape(fields), ",")
    Close #f
End Sub

' Parse a CSV line into fields (supports quotes and doubled "" inside)
Private Function CsvSplitToArray(ByVal s As String) As String()
    Dim cols As New Collection
    Dim i As Long, ch As String, token As String, inQ As Boolean
    i = 1
    Do While i <= Len(s)
        ch = mid$(s, i, 1)
        If ch = """" Then
            If inQ Then
                If i < Len(s) And mid$(s, i + 1, 1) = """" Then
                    token = token & """": i = i + 1
                Else
                    inQ = False
                End If
            Else
                inQ = True
            End If
        ElseIf ch = "," And Not inQ Then
            cols.Add token: token = ""
        Else
            token = token & ch
        End If
        i = i + 1
    Loop
    cols.Add token

    Dim out() As String: ReDim out(0 To cols.count - 1)
    For i = 1 To cols.count
        out(i - 1) = Replace$(cols(i), vbCr, "")
    Next i
    CsvSplitToArray = out
End Function

' Locale-safe parse of "yyyy-mm-dd hh:nn:ss"
Private Function ParseIsoDateTime(ByVal s As String) As Date
    On Error Resume Next
    If Len(s) >= 19 Then
        Dim yr As Long, mo As Long, dy As Long, hh As Long, nn As Long, ss As Long
        yr = CLng(mid$(s, 1, 4)): mo = CLng(mid$(s, 6, 2)): dy = CLng(mid$(s, 9, 2))
        hh = CLng(mid$(s, 12, 2)): nn = CLng(mid$(s, 15, 2)): ss = CLng(mid$(s, 18, 2))
        ParseIsoDateTime = DateSerial(yr, mo, dy) + TimeSerial(hh, nn, ss)
    Else
        ParseIsoDateTime = 0
    End If
End Function

Private Function CsvEscape(ByVal fields As Variant) As Variant
    Dim i As Long, n As Long
    n = UBound(fields) - LBound(fields) + 1
    Dim out() As String: ReDim out(0 To n - 1)
    For i = 0 To n - 1
        Dim v As String: v = CStr(fields(LBound(fields) + i))
        If InStr(v, """") > 0 Then v = Replace$(v, """", """""")
        If InStr(v, ",") > 0 Or InStr(v, vbCr) > 0 Or InStr(v, vbLf) > 0 Or _
           Left$(v, 1) = " " Or Right$(v, 1) = " " Then
            v = """" & v & """"
        End If
        out(i) = v
    Next i
    CsvEscape = out
End Function

'===============================================================================
'=== ATTACHMENT EXTRACTION (unique, newest-wins + CSV log + inline heuristics) =
'===============================================================================
Public Function ExtractConversationAttachments(ByVal sortedDesc As Collection, _
                                               ByVal convDir As String) As String
    On Error GoTo EH

    Dim attDir As String: attDir = EnsureFolderDeep(AppendPath(convDir, ATT_SUBDIR))
    Dim logPath As String: logPath = EnsureCsvLog(attDir)

    ' Load manifest (filename -> last attachment_mod_time we saved)
    Dim manifest As Object: Set manifest = LoadCsvManifest(logPath)

    Dim winners As Object ' key: LCase(filename) -> Array(modTime As Date, att As Outlook.Attachment, mi As Outlook.MailItem, fileName As String, isInline As Boolean, sizeBytes As Long)
    Set winners = CreateObject("Scripting.Dictionary")

    Dim i As Long, attIdx As Long
    Dim mi As Outlook.MailItem
    Dim att As Outlook.Attachment

    ' Pass 1: choose newest per filename (after filtering)
    For i = 1 To sortedDesc.count
        Set mi = sortedDesc(i)
        If Not mi Is Nothing Then
            For attIdx = 1 To mi.attachments.count
                Set att = mi.attachments(attIdx)
                If ShouldConsiderAttachment(att) Then
                    Dim fname As String: fname = BestSafeFileName(att, attIdx)
                    If Len(fname) > 0 Then
                        Dim key As String: key = LCase$(fname)
                        Dim modt As Date: modt = GetAttachmentModTime(att, mi)
                        Dim inl As Boolean: inl = IsInlineAttachment(att)
                        Dim sz As Long: sz = att.size

                        If Not winners.Exists(key) Then
                            winners.Add key, Array(modt, att, mi, fname, inl, sz)
                        Else
                            Dim cur As Variant: cur = winners(key)
                            If modt > cur(0) Then winners(key) = Array(modt, att, mi, fname, inl, sz)
                        End If
                    End If
                End If
            Next attIdx
        End If
    Next i

    ' Pass 2: save winners (overwrite only if newer) + CSV row
    Dim fso As Object: Set fso = CreateObject("Scripting.FileSystemObject")
    Dim savedNew As Long, overwritten As Long, skippedOlder As Long
    Dim keys As Variant: keys = winners.keys

    For i = LBound(keys) To UBound(keys)
        Dim rec As Variant: rec = winners(keys(i))
        Dim bestMod As Date: bestMod = rec(0)
        Set att = rec(1)
        Set mi = rec(2)
        ' preserve extension when trimming to budget, add short hash if truncated
        Dim targetName As String: targetName = FitFileNameForBudgetPreserveExt(rec(3), attDir, 6)
        Dim fullPath As String: fullPath = AppendPath(attDir, targetName)
        Dim wasInline As Boolean: wasInline = rec(4)
        Dim sizeBytes As Long: sizeBytes = rec(5)

        ' Determine the existing timestamp to compare:
        Dim existingMod As Date: existingMod = 0
        If manifest.Exists(LCase$(targetName)) Then
            existingMod = ParseIsoDateTime(CStr(manifest(LCase$(targetName))))
        ElseIf fso.fileExists(fullPath) Then
            existingMod = fso.GetFile(fullPath).DateLastModified
        End If

        Dim action As String
        Dim ok As Boolean

        If fso.fileExists(fullPath) Then
            If bestMod > existingMod Then
                On Error Resume Next
                att.SaveAsFile fullPath
                ok = (Err.Number = 0)
                Err.Clear
                On Error GoTo 0
                If ok Then
                    action = "overwritten_newer"
                    overwritten = overwritten + 1
                    manifest(LCase$(targetName)) = Format$(bestMod, "yyyy-mm-dd hh:nn:ss")
                Else
                    action = "error_overwrite"
                End If
            Else
                action = "skipped_older"
                skippedOlder = skippedOlder + 1
            End If
        Else
            On Error Resume Next
            att.SaveAsFile fullPath
            ok = (Err.Number = 0)
            Err.Clear
            On Error GoTo 0
            If ok Then
                action = "saved_new"
                savedNew = savedNew + 1
                manifest(LCase$(targetName)) = Format$(bestMod, "yyyy-mm-dd hh:nn:ss")
            Else
                action = "error_save"
            End If
        End If

        ' CSV row (ISO date/time for run_date/run_time)
        AppendCsvRow logPath, Array( _
            Format$(Now, "yyyy-mm-dd"), _
            Format$(Now, "HH:nn:ss"), _
            action, _
            targetName, _
            CStr(sizeBytes), _
            IIf(wasInline, "inline", "regular"), _
            SafeGetSenderSmtp(mi), _
            MakeSmartSubject(NullToEmpty(mi.subject)), _
            Format$(EffectiveLocalTime(mi), "yyyy-mm-dd hh:nn:ss"), _
            Format$(bestMod, "yyyy-mm-dd hh:nn:ss"), _
            NullToEmpty(mi.entryId), _
            fullPath _
        )
    Next i

    ExtractConversationAttachments = "attachments: " & savedNew & " saved, " & _
                                     overwritten & " overwritten (newer), " & _
                                     skippedOlder & " skipped (older). Log: " & logPath
    Exit Function
EH:
    ExtractConversationAttachments = "attachments: error (" & Err.Number & "): " & Err.Description
End Function

' Include / exclude rules (inline filtering)
Private Function ShouldConsiderAttachment(ByVal att As Outlook.Attachment) As Boolean
    On Error Resume Next
    If att Is Nothing Then Exit Function

    Dim pa As Object: Set pa = att.PropertyAccessor

    ' Hidden attachments
    Dim isHidden As Variant: isHidden = pa.GetProperty(PR_ATTACHMENT_HIDDEN)
    If VarType(isHidden) = vbBoolean Then If isHidden Then Exit Function

    ' Name & MIME-based exclusions
    Dim fname As String: fname = LCase$(Trim$(NullToEmpty(att.fileName)))
    If fname = "" Then fname = "attachment"
    If fname = "winmail.dat" Then Exit Function

    Dim mimeTag As String: mimeTag = LCase$(NullToEmpty(pa.GetProperty(PR_ATTACH_MIME_TAG)))
    If mimeTag = "application/pkcs7-signature" Or mimeTag = "application/x-pkcs7-signature" Then Exit Function ' smime.p7s

    ' Skip nested message wrappers; processed separately and not retained
    If att.Type = OL_ATTACHMENTTYPE_EMBEDDED Then Exit Function
    If Right$(fname, 4) = ".msg" Or Right$(fname, 4) = ".eml" Then Exit Function

    Dim sizeBytes As Long: sizeBytes = att.size

    ' Inline classification
    Dim isInline As Boolean: isInline = IsInlineAttachment(att)

    ' Signature/logo/tracking heuristics (skip)
    If IsLikelySignatureAsset(att, pa, fname, sizeBytes, isInline) Then Exit Function

    If isInline Then
        ' Only keep inline if it looks like user-uploaded content
        ShouldConsiderAttachment = INCLUDE_INLINE_UPLOADED And (sizeBytes >= MIN_INLINE_UPLOAD_BYTES)
        Exit Function
    End If

    ' Regular attachment -> keep
    ShouldConsiderAttachment = True
End Function

' Inline if Content-Disposition=inline or has a Content-ID
Private Function IsInlineAttachment(ByVal att As Outlook.Attachment) As Boolean
    On Error Resume Next
    Dim pa As Object: Set pa = att.PropertyAccessor
    Dim disp As String: disp = LCase$(NullToEmpty(pa.GetProperty(PR_ATTACH_CONTENT_DISPOSITION)))
    If disp = "inline" Then IsInlineAttachment = True: Exit Function
    Dim cid As String: cid = Trim$(NullToEmpty(pa.GetProperty(PR_ATTACH_CONTENT_ID)))
    If Len(cid) > 0 Then IsInlineAttachment = True
End Function

' Heuristic to drop signature/logo assets, social icons, tracking pixels, etc.
Private Function IsLikelySignatureAsset(ByVal att As Outlook.Attachment, ByVal pa As Object, _
                                        ByVal fname As String, ByVal sizeBytes As Long, _
                                        ByVal isInline As Boolean) As Boolean
    On Error Resume Next

    Dim base As String: base = fname
    Dim dotPos As Long: dotPos = InStrRev(base, ".")
    Dim stem As String, ext As String
    If dotPos > 0 Then
        stem = Left$(base, dotPos - 1)
        ext = LCase$(mid$(base, dotPos + 1))
    Else
        stem = base
        ext = ""
    End If

    ' Small inline images (logos/social icons)
    If isInline And sizeBytes <= MAX_SIGNATURE_IMAGE_BYTES Then
        IsLikelySignatureAsset = True
        Exit Function
    End If

    ' Name patterns often used by signatures
    Dim lowStem As String: lowStem = LCase$(stem)
    If lowStem Like "image0#" Or lowStem Like "image##" Or lowStem Like "image###" Then
        IsLikelySignatureAsset = True: Exit Function
    End If
    If lowStem Like "*logo*" Or lowStem Like "*signature*" Or lowStem Like "*sig*" Then
        IsLikelySignatureAsset = True: Exit Function
    End If
    If lowStem Like "*facebook*" Or lowStem Like "*linkedin*" Or lowStem Like "*twitter*" Or _
       lowStem Like "*instagram*" Or lowStem Like "*youtube*" Or lowStem Like "*whatsapp*" Or _
       lowStem Like "*icon*" Or lowStem Like "*badge*" Then
        IsLikelySignatureAsset = True: Exit Function
    End If

    ' Tracking pixel by MIME/size
    Dim mimeTag As String: mimeTag = LCase$(NullToEmpty(pa.GetProperty(PR_ATTACH_MIME_TAG)))
    If (mimeTag Like "image/*" And sizeBytes < 1024) Then
        IsLikelySignatureAsset = True: Exit Function
    End If

    ' Vector/brand assets common in signatures
    If ext = "svg" Or ext = "ico" Or ext = "wmf" Then
        IsLikelySignatureAsset = True: Exit Function
    End If

    IsLikelySignatureAsset = False
End Function

' Compute the best "last modified" timestamp for the attachment (intrinsic)
Private Function GetAttachmentModTime(ByVal att As Outlook.Attachment, ByVal parentMail As Outlook.MailItem) As Date
    On Error Resume Next
    Dim pa As Object: Set pa = att.PropertyAccessor
    Dim d As Date: d = pa.GetProperty(PR_LAST_MODIFICATION_TIME)
    If d = 0 Then d = pa.GetProperty(PR_CREATION_TIME)
    If d = 0 Then d = EffectiveLocalTime(parentMail)
    If d = 0 Then d = Now
    GetAttachmentModTime = d
End Function

' Safe filename with minimal sanitation (keeps extension)
Private Function BestSafeFileName(ByVal att As Outlook.Attachment, ByVal fallbackIndex As Long) As String
    On Error Resume Next
    Dim nm As String: nm = Trim$(NullToEmpty(att.fileName))
    If Len(nm) = 0 Then nm = "attachment_" & CStr(fallbackIndex)
    BestSafeFileName = SanitizeName(nm)
End Function

'===============================================================================
'=== NESTED MESSAGES ============================================================
'===============================================================================

' Write a nested conversation (helper used by both nested flows)
Private Function WriteConversationFromMail(ByVal seed As Outlook.MailItem, ByVal convDir As String) As String
    On Error GoTo EH
    If seed Is Nothing Then Exit Function
    If IsDraftMail(seed) Then Exit Function

    ' Include Sent Items across stores so nested originals are captured too
    Dim convMsgs As Collection
    Set convMsgs = CollectConversationMailsIncludingSent(seed)
    If convMsgs Is Nothing Or convMsgs.count = 0 Then Exit Function

    Dim nAsc As Collection
    Dim nDesc As Collection
    Set nAsc = SortMailsByDateAsc(convMsgs)
    Set nDesc = SortMailsByDateDesc(convMsgs)

    Dim init As Outlook.MailItem
    Dim newest As Outlook.MailItem
    Set init = nAsc(1)
    Set newest = nDesc(1)

    Dim smartSub As String
    smartSub = MakeSmartSubject(NullToEmpty(newest.subject))
    Dim subjFS As String
    subjFS = SanitizeName(smartSub)
    ' Stable conv_ref in nested, too (header only; file name already independent)
    Dim nRefRaw As String
    nRefRaw = ComputeStableConvRef(init, convMsgs)

    Dim base As String
    base = FitForPathBudget("conversation_" & subjFS & NESTED_SUFFIX, convDir, ".txt")

    Dim nestedTxtRag As String
    Dim nestedTxtHuman As String
    nestedTxtRag = AppendPath(convDir, base & ".txt")
    nestedTxtHuman = AppendPath(convDir, base & "_human.txt")

    Dim ok As Boolean
    ' Use the earliest message as the "initial" one for header To/Cc display
    ok = WriteConversationFileRag(nestedTxtRag, nDesc, nRefRaw, smartSub, NullToEmpty(init.entryId))
    If ok Then
        Dim okHuman As Boolean
        okHuman = WriteConversationFileHuman(nestedTxtHuman, nDesc, nRefRaw, smartSub, NullToEmpty(init.entryId))
        If Not okHuman Then
            Trace "  WARN: failed to write nested human conversation file: " & nestedTxtHuman
        End If
    End If

    Dim ignored As String
    ignored = ExtractConversationAttachments(nDesc, convDir)

    WriteConversationFromMail = nestedTxtRag
    Exit Function

EH:
    WriteConversationFromMail = ""
End Function

Public Function ProcessNestedMessages(ByVal mainSortedDesc As Collection, ByVal convDir As String) As String
    On Error GoTo EH

    Dim mainKeys As Object: Set mainKeys = CreateObject("Scripting.Dictionary"): mainKeys.CompareMode = 1
    Dim i As Long
    For i = 1 To mainSortedDesc.count
        Dim mi0 As Outlook.MailItem: Set mi0 = mainSortedDesc(i)
        If Not mi0 Is Nothing Then
            Dim convKey As String: convKey = GetConvKey(mi0)
            If Len(convKey) > 0 Then mainKeys(convKey) = True
        End If
    Next i

    Dim visitedConv As Object: Set visitedConv = CreateObject("Scripting.Dictionary"): visitedConv.CompareMode = 1
    Dim created As New Collection, tmpDeleted As Long, nestedCount As Long

    For i = 1 To mainSortedDesc.count
        Dim mi As Outlook.MailItem: Set mi = mainSortedDesc(i)
        If Not mi Is Nothing Then
            Call ProcessNestedFromMail(mi, convDir, mainKeys, visitedConv, created, tmpDeleted, nestedCount, 1, PROCESS_NESTED_DEPTH_MAX)
        End If
    Next i

    Dim summary As String
    summary = "nested (attachments walk): " & nestedCount & " conversation(s); temp files cleaned: " & tmpDeleted
    If created.count > 0 Then
        summary = summary & vbCrLf & "nested conv files:" & vbCrLf
        Dim limit As Long: limit = 6
        Dim j As Long
        For j = 1 To created.count
            If j > limit Then summary = summary & "...": Exit For
            summary = summary & created(j) & vbCrLf
        Next j
    End If

    ProcessNestedMessages = summary
    Exit Function
EH:
    ProcessNestedMessages = "nested (attachments walk): error (" & Err.Number & "): " & Err.Description
End Function

Private Sub ProcessNestedFromMail(ByVal mail As Outlook.MailItem, ByVal convDir As String, _
                                  ByVal mainKeys As Object, ByVal visitedConv As Object, _
                                  ByRef created As Collection, ByRef tmpDeleted As Long, _
                                  ByRef nestedCount As Long, ByVal depth As Long, ByVal maxDepth As Long)
    On Error Resume Next
    If mail Is Nothing Then Exit Sub
    If depth > maxDepth Then Exit Sub

    Dim attIdx As Long
    For attIdx = 1 To mail.attachments.count
        Dim att As Outlook.Attachment: Set att = mail.attachments(attIdx)
        If LooksLikeNestedMessage(att) Then
            Dim tmp As String: tmp = ""
            Dim nm As Outlook.MailItem: Set nm = OpenAttachmentAsMail(att, tmp)
            If Not nm Is Nothing Then
                Dim nk As String: nk = GetConvKey(nm)
                If Len(nk) > 0 And Not mainKeys.Exists(nk) And Not visitedConv.Exists(nk) Then
                    visitedConv(nk) = True
                    nestedCount = nestedCount + 1
                    Dim pth As String: pth = WriteConversationFromMail(nm, convDir)
                    If Len(pth) > 0 Then created.Add pth
                    ProcessNestedFromMail nm, convDir, mainKeys, visitedConv, created, tmpDeleted, nestedCount, depth + 1, maxDepth
                End If
            End If
            If Len(tmp) > 0 Then
                Dim fso As Object: Set fso = CreateObject("Scripting.FileSystemObject")
                On Error Resume Next
                fso.DeleteFile tmp, True
                If Err.Number = 0 Then tmpDeleted = tmpDeleted + 1
                Err.Clear: On Error GoTo 0
            End If
        End If
    Next attIdx
End Sub

Private Function LooksLikeNestedMessage(ByVal att As Outlook.Attachment) As Boolean
    On Error Resume Next
    If att Is Nothing Then Exit Function
    Dim fname As String: fname = LCase$(Trim$(NullToEmpty(att.fileName)))
    If Right$(fname, 4) = ".msg" Or Right$(fname, 4) = ".eml" Then LooksLikeNestedMessage = True: Exit Function
    If att.Type = OL_ATTACHMENTTYPE_EMBEDDED Then
        Dim obj As Object: Set obj = att.EmbeddedItem
        If TypeName(obj) = "MailItem" Then LooksLikeNestedMessage = True
    End If
End Function

Private Function OpenAttachmentAsMail(ByVal att As Outlook.Attachment, ByRef tmpPathOut As String) As Outlook.MailItem
    On Error Resume Next
    tmpPathOut = ""
    If att Is Nothing Then Exit Function

    ' Embedded Outlook item (no temp file needed)
    If att.Type = OL_ATTACHMENTTYPE_EMBEDDED Then
        Dim obj As Object: Set obj = att.EmbeddedItem
        If TypeName(obj) = "MailItem" Then Set OpenAttachmentAsMail = obj
        Exit Function
    End If

    ' Save .msg/.eml to temp and open
    Dim fname As String: fname = LCase$(NullToEmpty(att.fileName))
    If Right$(fname, 4) = ".msg" Or Right$(fname, 4) = ".eml" Then
        Dim tmpDir As String: tmpDir = GetTempWorkDir()
        Dim saveAs As String
        If Len(att.fileName) > 0 Then
            saveAs = AppendPath(tmpDir, SanitizeName(att.fileName))
        Else
            saveAs = AppendPath(tmpDir, "nested_" & Format$(Now, "yyyymmddhhnnss") & ".msg")
        End If

        Dim fso As Object: Set fso = CreateObject("Scripting.FileSystemObject")
        If fso.fileExists(saveAs) Then
            Dim base As String, ext As String, dot As Long
            dot = InStrRev(saveAs, ".")
            If dot > 0 Then base = Left$(saveAs, dot - 1): ext = mid$(saveAs, dot) Else base = saveAs: ext = ""
            Dim idx As Long: idx = CLng(Timer * 1000) And &HFFFF&
            saveAs = base & "_" & CStr(idx) & ext
        End If

        On Error Resume Next
        att.SaveAsFile saveAs
        If Err.Number <> 0 Then Err.Clear: Exit Function
        On Error GoTo 0

        Dim sess As Outlook.NameSpace: Set sess = Application.Session
        Dim o As Object: Set o = sess.OpenSharedItem(saveAs)
        If TypeName(o) = "MailItem" Then
            Set OpenAttachmentAsMail = o
            tmpPathOut = saveAs
        Else
            On Error Resume Next
            fso.DeleteFile saveAs, True
            On Error GoTo 0
        End If
    End If
End Function

Private Function GetTempWorkDir() As String
    On Error Resume Next
    Dim d As String
    d = Environ$("TEMP")
    If Len(d) = 0 Then d = Environ$("TMP")
    If Len(d) = 0 Then d = "C:\Temp"
    GetTempWorkDir = EnsureFolderDeep(AppendPath(d, "ConvTxt_Work"))
End Function

' Conversation key builder (ConversationID preferred, else normalized subject/topic)
Private Function GetConvKey(ByVal mi As Outlook.MailItem) As String
    On Error Resume Next
    Dim k As String
    k = NullToEmpty(mi.conversationID)
    If Len(k) > 0 Then
        GetConvKey = k
    Else
        Dim t As String: t = NullToEmpty(mi.ConversationTopic)
        If Len(t) = 0 Then t = NullToEmpty(mi.subject)
        GetConvKey = LCase$(MakeSmartSubject(t))
    End If
End Function

' Build a set (Dictionary) of conversation keys for a collection of mails
Private Function BuildConvIdSet(ByVal msgs As Collection) As Object
    Dim dict As Object: Set dict = CreateObject("Scripting.Dictionary")
    dict.CompareMode = 1 ' TextCompare
    Dim i As Long
    For i = 1 To msgs.count
        Dim mi As Outlook.MailItem: Set mi = msgs(i)
        If Not mi Is Nothing Then
            Dim cid As String: cid = GetConvKey(mi)
            If Len(cid) > 0 Then dict(cid) = True
        End If
    Next i
    Set BuildConvIdSet = dict
End Function

' Process .msg/.eml already present in attachments folder
Public Function ProcessNestedMessageAttachmentsInFolder(ByVal mainSortedDesc As Collection, ByVal convDir As String) As String
    On Error GoTo EH

    Dim attDir As String: attDir = EnsureFolderDeep(AppendPath(convDir, ATT_SUBDIR))
    Dim fso As Object: Set fso = CreateObject("Scripting.FileSystemObject")

    Dim mainKeys As Object: Set mainKeys = CreateObject("Scripting.Dictionary"): mainKeys.CompareMode = 1
    Dim i As Long
    For i = 1 To mainSortedDesc.count
        Dim mi0 As Outlook.MailItem: Set mi0 = mainSortedDesc(i)
        If Not mi0 Is Nothing Then
            Dim convKey As String: convKey = GetConvKey(mi0)
            If Len(convKey) > 0 Then mainKeys(convKey) = True
        End If
    Next i

    Dim visited As Object: Set visited = CreateObject("Scripting.Dictionary"): visited.CompareMode = 1
    Dim processed As Long, deleted As Long, errs As Long

    Dim pass As Long
    For pass = 1 To NESTED_FOLDER_SCAN_PASSES
        If Not fso.FolderExists(attDir) Then Exit For

        Dim list As New Collection
        Dim fld As Object: Set fld = fso.GetFolder(attDir)
        Dim fi As Object
        For Each fi In fld.Files
            Dim nm As String
            nm = LCase$(fso.GetFileName(fi.path))
            ' Treat everything in attachments/ as an attachment, except our CSV log
            If nm <> "attachments_log.csv" Then
                list.Add fi.path
            End If
        Next fi

        If list.count = 0 Then Exit For

        Dim processedInPass As Long

        Dim p As Variant
        For Each p In list
            Dim itm As Object: Set itm = Nothing
            On Error Resume Next
            Set itm = Application.Session.OpenSharedItem(CStr(p))
            On Error GoTo 0

            If Not itm Is Nothing And TypeName(itm) = "MailItem" Then
                Dim nk As String: nk = GetConvKey(itm)
                If Len(nk) > 0 And Not mainKeys.Exists(nk) And Not visited.Exists(nk) Then
                    visited(nk) = True
                    Dim nestedTxt As String: nestedTxt = WriteConversationFromMail(itm, convDir)
                    If Len(nestedTxt) > 0 Then processed = processed + 1: processedInPass = processedInPass + 1
                    On Error Resume Next
                    fso.DeleteFile CStr(p), True
                    If Err.Number = 0 Then
                        deleted = deleted + 1
                        Dim logPath As String: logPath = EnsureCsvLog(attDir)
                        AppendCsvRow logPath, Array( _
                            Format$(Now, "yyyy-mm-dd"), Format$(Now, "HH:nn:ss"), "deleted_nested_msg", fso.GetFileName(CStr(p)), _
                            "0", "nested_msg", "(n/a)", "(n/a)", "", "", "", CStr(p) _
                        )
                    Else
                        errs = errs + 1
                    End If
                    Err.Clear: On Error GoTo 0
                End If
            Else
                errs = errs + 1
            End If
        Next p

        If processedInPass = 0 Then Exit For
    Next pass

    ProcessNestedMessageAttachmentsInFolder = "nested (folder scan): " & processed & " processed; " & deleted & " removed (.msg/.eml); " & errs & " errors."
    Exit Function
EH:
    ProcessNestedMessageAttachmentsInFolder = "nested (folder scan): error (" & Err.Number & "): " & Err.Description
End Function

'=== PUBLIC CONVERSATION PROCESSOR =============================================

Private Sub CleanupFailedConversationFolder(ByVal convDir As String)
    On Error Resume Next
    If Len(convDir) = 0 Then Exit Sub

    Dim fsoEH As Object
    Set fsoEH = CreateObject("Scripting.FileSystemObject")

    Dim pTxtEH As String
    ' Canonical name is "Conversation.txt"
    pTxtEH = AppendPath(convDir, "Conversation.txt")

    If Not fsoEH.fileExists(pTxtEH) And fsoEH.FolderExists(convDir) Then
        fsoEH.DeleteFolder convDir, True
    End If
End Sub

Public Function ProcessConversationIfNew(ByVal seed As Outlook.MailItem, _
                                         ByVal processedConvKeys As Object) As String
    Dim convDir As String
    On Error GoTo EH

    ProcessConversationIfNew = InternalProcessConversationIfNew(seed, processedConvKeys, convDir)
    Exit Function

EH:
    CleanupFailedConversationFolder convDir
    ProcessConversationIfNew = "SKIP: error (" & Err.Number & "): " & Err.Description
End Function

' Main implementation
Private Function InternalProcessConversationIfNew(ByVal seed As Outlook.MailItem, _
                                                  ByVal processedConvKeys As Object, _
                                                  ByRef convDir As String) As String
    Dim convMsgs As Collection
    Dim convKeys As Object
    Dim sig As String
    Dim k As Variant
    Dim sortedAsc As Collection
    Dim sortedDesc As Collection
    Dim initialMail As Outlook.MailItem
    Dim newestMail As Outlook.MailItem
    Dim smartSubj As String
    Dim convRefRaw As String
    Dim convRefFS As String
    Dim subjFS As String
    Dim outRoot As String
    Dim folderName As String
    Dim pTxtRag As String
    Dim pTxtHuman As String
    Dim okWriteRag As Boolean
    Dim okWriteHuman As Boolean
    Dim attSummary As String
    Dim nestedSummary1 As String
    Dim nestedSummary2 As String
    Dim ck As Variant

    convDir = ""

    If seed Is Nothing Then
        InternalProcessConversationIfNew = "SKIP: (no seed)"
        Exit Function
    End If

    ' Exclude drafts as seeds
    If IsDraftMail(seed) Then
        InternalProcessConversationIfNew = "SKIP: draft / unsent message."
        Exit Function
    End If

    ' 1) Build the conversation from the seed (includes Sent Items across stores)
    Set convMsgs = CollectConversationMailsIncludingSent(seed)
    If convMsgs Is Nothing Or convMsgs.count = 0 Then
        InternalProcessConversationIfNew = "SKIP: Could not enumerate the conversation."
        Exit Function
    End If

    ' 2) Build keys for this conversation (ID preferred; smart-subject fallback)
    Set convKeys = BuildConvIdSet(convMsgs)

    ' Stable conversation signature by sorted Internet Message-IDs (fallback when absent)
    sig = BuildStableIdSignature(convMsgs)

    ' If ANY key was already processed this run, skip entirely (including signature)
    For Each k In convKeys.keys
        If processedConvKeys.Exists(CStr(k)) Then
            InternalProcessConversationIfNew = "SKIP: conversation already processed in this run (ConvKey)."
            Exit Function
        End If
    Next k
    If processedConvKeys.Exists(sig) Then
        InternalProcessConversationIfNew = "SKIP: conversation already processed in this run (signature)."
        Exit Function
    End If

    ' 3) Sort, compute naming
    Set sortedAsc = SortMailsByDateAsc(convMsgs)    ' oldest..newest
    Set sortedDesc = SortMailsByDateDesc(convMsgs)  ' newest..oldest

    Set initialMail = sortedAsc(1)
    Set newestMail = sortedDesc(1)

    smartSubj = MakeSmartSubject(NullToEmpty(newestMail.subject))
    ' Stable reference: earliest message date + 6-hex fingerprint of conv key; TML/EML if ANY message includes FLAG_ADDR
    convRefRaw = ComputeStableConvRef(initialMail, convMsgs)

    convRefFS = SanitizeName(convRefRaw)
    subjFS = SanitizeName(smartSubj)

    ' 4) Ensure output folder for this conversation (picker-aware)
    outRoot = GetRunOutRoot()

    ' Reserve budget for attachments so filenames aren't forced to 2–3 chars
    Dim extraAttach As String
    Dim extraConv As String
    Dim extraSuffix As String

    extraAttach = "\attachments\" & String$(RESERVED_ATTACHMENT_FILENAME_CHARS, "X")
    extraConv = "\Conversation.txt"
    If Len(extraAttach) > Len(extraConv) Then
        extraSuffix = extraAttach
    Else
        extraSuffix = extraConv
    End If

    folderName = FitForPathBudget(convRefFS & " - " & subjFS, outRoot, extraSuffix)
    convDir = EnsureFolderDeep(AppendPath(outRoot, folderName))

    ' 5) Write Conversation.txt (RAG) + Conversation_human.txt
    pTxtRag = AppendPath(convDir, "Conversation.txt")
    pTxtHuman = AppendPath(convDir, "Conversation_human.txt")

    Trace "About to write: " & pTxtRag & " | conv_ref=" & convRefRaw
    okWriteRag = WriteConversationFileRag(pTxtRag, sortedDesc, convRefRaw, smartSubj, NullToEmpty(initialMail.entryId))
    Trace "WriteConversationFileRag returned: " & CStr(okWriteRag)

    If Not okWriteRag Then
        TryDeleteFolderIfEmpty convDir
        InternalProcessConversationIfNew = "SKIP: failed to write conversation file."
        Exit Function
    End If

    ' Human version is best-effort – failure shouldn't nuke the RAG export
    okWriteHuman = WriteConversationFileHuman(pTxtHuman, sortedDesc, convRefRaw, smartSubj, NullToEmpty(initialMail.entryId))
    If Not okWriteHuman Then
        Trace "  WARN: failed to write Conversation_human.txt (non-fatal)."
    End If

    ' 6) Attachments + nested + cleanup
    attSummary = ExtractConversationAttachments(sortedDesc, convDir)
    nestedSummary1 = ProcessNestedMessages(sortedDesc, convDir)
    nestedSummary2 = ProcessNestedMessageAttachmentsInFolder(sortedDesc, convDir)

    ' 6.5) Write manifest.json + global conversations_index.jsonl (idempotent)
    MI_WriteManifestAndIndex convDir, convMsgs, sortedDesc, convRefRaw, smartSubj, sig, _
                             GetRunOutRoot(), pTxtRag

    ' 7) Mark ALL keys of this conversation as processed for this run (plus signature)
    For Each ck In convKeys.keys
        processedConvKeys(ck) = True
    Next ck
    processedConvKeys(sig) = True

    ' 8) Return a concise summary line
    InternalProcessConversationIfNew = pTxtRag & vbCrLf & attSummary & vbCrLf & nestedSummary1 & vbCrLf & nestedSummary2
End Function

'========================
' Sent-Items union helpers
'========================

' Escape AQS literal for Items.Restrict
Private Function RestrictLiteral(ByVal s As String) As String
    RestrictLiteral = Replace$(s, "'", "''")
End Function

' My SMTP (used to ensure we only add our own outbound from Sent Items)
Private Function GetCurrentUserSmtp() As String
    On Error Resume Next
    Dim ae As Outlook.AddressEntry
    Set ae = Application.Session.CurrentUser.AddressEntry
    GetCurrentUserSmtp = NormalizeSmtpAddress(ae)
    If Len(GetCurrentUserSmtp) = 0 Then
        GetCurrentUserSmtp = LCase$(NullToEmpty(Application.Session.CurrentUser.Address))
    End If
End Function

' Collect conversation via Outlook.Conversation AND union with Sent Items across ALL stores.
Private Function CollectConversationMailsIncludingSent(ByVal seed As Outlook.MailItem) As Collection
    Dim base As Collection
    Set base = CollectConversationMails(seed) ' existing logic

    ' Build seen-set keyed by EntryID or InternetMessageID to avoid dups
    Dim seen As Object: Set seen = CreateObject("Scripting.Dictionary"): seen.CompareMode = 1
    Dim i As Long
    For i = 1 To base.count
        Dim k As String: k = GetInternetMessageId(base(i))
        If Len(k) = 0 Then k = LCase$(NullToEmpty(base(i).entryId))
        If Len(k) = 0 Then k = CStr(base(i).CreationTime) & "|" & LCase$(NullToEmpty(base(i).subject))
        seen(k) = True
    Next i

    Dim topic As String: topic = LCase$(MakeSmartSubject(NullToEmpty(seed.ConversationTopic)))
    If Len(topic) = 0 Then topic = LCase$(MakeSmartSubject(NullToEmpty(seed.subject)))
    Dim convId As String: convId = NullToEmpty(seed.conversationID)
    Dim mySmtp As String: mySmtp = GetCurrentUserSmtp()

    ' Harvest Sent Items across EVERY store
    HarvestSentAcrossStoresByConversation base, seen, convId, topic, mySmtp

    Set CollectConversationMailsIncludingSent = base
End Function

Private Sub HarvestSentAcrossStoresByConversation(ByRef out As Collection, ByRef seen As Object, _
                                                  ByVal convId As String, ByVal topic As String, _
                                                  ByVal mySmtp As String)
    On Error Resume Next
    Dim st As Outlook.store, f As Outlook.MAPIFolder, itms As Outlook.items, r As Outlook.items
    Dim ns As Outlook.NameSpace: Set ns = Application.Session
    Dim filter As String, t As String: t = RestrictLiteral(topic)

    ' Filter by ConversationTopic first (broad), then verify in code.
    filter = "[MessageClass] = 'IPM.Note' AND [ConversationTopic] = '" & t & "'"

    For Each st In ns.Stores
        Set f = Nothing
        Err.Clear
        Set f = st.GetDefaultFolder(olFolderSentMail)
        If Err.Number <> 0 Or f Is Nothing Then GoTo nextStore

        Set itms = f.items
        If itms Is Nothing Then GoTo nextStore

        itms.Sort "[SentOn]", True
        Set r = itms.Restrict(filter)

        Dim obj As Object
        For Each obj In r
            If TypeName(obj) = "MailItem" Then
                Dim mi As Outlook.MailItem: Set mi = obj

                ' Verify: same conversation (ID or normalized topic) AND it's from me
                Dim okConv As Boolean
                okConv = (Len(convId) > 0 And StrComp(convId, NullToEmpty(mi.conversationID), vbBinaryCompare) = 0)
                If Not okConv Then
                    okConv = (LCase$(MakeSmartSubject(NullToEmpty(mi.ConversationTopic))) = topic)
                End If

                Dim fromMe As Boolean: fromMe = (LCase$(SafeGetSenderSmtp(mi)) = LCase$(mySmtp))

                If okConv And fromMe Then
                    If Not IsDraftMail(mi) Then
                        Dim k As String
                        k = GetInternetMessageId(mi)
                        If Len(k) = 0 Then k = LCase$(NullToEmpty(mi.entryId))
                        If Len(k) = 0 Then k = CStr(mi.CreationTime) & "|" & LCase$(NullToEmpty(mi.subject))
                        If Not seen.Exists(k) Then
                            out.Add mi
                            seen(k) = True
                        End If
                    End If
                End If
            End If
        Next obj
nextStore:
    Next st
End Sub

'===============================================================================
' manifest.json + conversations_index.jsonl (canonical v1 manifest)
'===============================================================================
Private Sub MI_WriteManifestAndIndex( _
    ByVal convDir As String, _
    ByVal convMsgs As Collection, _
    ByVal sortedDesc As Collection, _
    ByVal convRefRaw As String, _
    ByVal smartSubj As String, _
    ByVal runSignature As String, _
    ByVal outRoot As String, _
    ByVal conversationTxtPath As String)

    On Error GoTo EH

    Dim fso As Object
    Set fso = CreateObject("Scripting.FileSystemObject")

    '--- 1) Time range ----------------------------------------------------------
    Dim tFirst As Date, tLast As Date
    Dim i As Long

    If convMsgs Is Nothing Or convMsgs.count = 0 Then Exit Sub

    tFirst = EffectiveLocalTime(convMsgs(1))
    tLast = tFirst

    For i = 1 To convMsgs.count
        Dim dt As Date
        dt = EffectiveLocalTime(convMsgs(i))
        If dt < tFirst Then tFirst = dt
        If dt > tLast Then tLast = dt
    Next i

    Dim startedIso As String
    Dim endedIso As String

    ' Canonical requires ISO-Z strings; we serialize local effective time in that shape.
    startedIso = FormatIsoUtcString(tFirst)
    endedIso = FormatIsoUtcString(tLast)

    '--- 2) Attachment count ----------------------------------------------------
    Dim attDir As String
    attDir = AppendPath(convDir, ATT_SUBDIR)

    Dim attachmentCount As Long
    attachmentCount = 0

    If fso.FolderExists(attDir) Then
        Dim fld As Object
        Dim fi As Object
        Set fld = fso.GetFolder(attDir)

        For Each fi In fld.Files
            Dim nm As String
            nm = LCase$(fso.GetFileName(fi.path))
            ' Treat everything in attachments/ as an attachment, except our CSV log
            If nm <> "attachments_log.csv" Then
                attachmentCount = attachmentCount + 1
            End If
        Next fi
    End If

    '--- 3) SHA-256 of Conversation.txt (best-effort; B1 may recompute) --------
    Dim txtSha As String
    If fso.fileExists(conversationTxtPath) Then
        txtSha = FileSha256HexNormalized(conversationTxtPath)
    Else
        txtSha = ""
    End If

    '--- 4) Build canonical manifest.json (v1) ---------------------------------
    Dim folderRel As String
    folderRel = GetFolderLeafName(convDir)

    Dim pathsJson As String
    pathsJson = "{""conversation_txt"":""Conversation.txt"",""attachments_dir"":""attachments/""}"

    Dim manifest As String
    manifest = "{""manifest_version"":""1""" & _
               ",""folder"":" & JsonQuote(folderRel) & _
               ",""subject_label"":" & JsonQuote(smartSubj) & _
               ",""message_count"":" & CStr(convMsgs.count) & _
               ",""started_at_utc"":" & JsonQuote(startedIso) & _
               ",""ended_at_utc"":" & JsonQuote(endedIso) & _
               ",""attachment_count"":" & CStr(attachmentCount) & _
               ",""paths"":" & pathsJson & _
               ",""sha256_conversation"":" & JsonQuote(txtSha) & "}"

    Dim manifestPath As String
    manifestPath = AppendPath(convDir, "manifest.json")

    Dim ok As Boolean
    ok = SaveUtf8Atomic(manifest, manifestPath & ".tmp", manifestPath)
    ' best effort; even if this fails, we must not break the main export flow

    '--- 5) Append to global conversations_index.jsonl idempotently ------------
    Dim idxPath As String
    idxPath = AppendPath(outRoot, "conversations_index.jsonl")

    Dim markerPath As String
    markerPath = AppendPath(convDir, ".indexed.sha256")

    ' If we couldn't compute SHA, skip idempotent indexing to avoid duplicates.
    If Len(Trim$(txtSha)) = 0 Then GoTo DONE

    Dim already As String
    already = ""
    If fso.fileExists(markerPath) Then
        already = ReadAllText(markerPath)
    End If

    If StrComp(LCase$(Trim$(already)), LCase$(Trim$(txtSha)), vbTextCompare) <> 0 Then
        Dim line As String

        ' Index format is your internal helper; no canonical constraints here.
        line = "{""conv_ref"":" & JsonQuote(convRefRaw) & _
               ",""subject"":" & JsonQuote(smartSubj) & _
               ",""time_first"":" & JsonQuote(Format$(tFirst, "yyyy-mm-dd\Thh:nn:ss")) & _
               ",""time_last"":" & JsonQuote(Format$(tLast, "yyyy-mm-dd\Thh:nn:ss")) & _
               ",""message_count"":" & CStr(convMsgs.count) & _
               ",""attachments_count"":" & CStr(attachmentCount) & _
               ",""txt_sha256"":" & JsonQuote(txtSha) & _
               ",""conversation_txt_path"":" & JsonQuote(conversationTxtPath) & "}"

        AppendTextLine idxPath, line

        ' Update marker (atomic best-effort)
        ok = SaveUtf8Atomic(txtSha, markerPath & ".tmp", markerPath)
    End If

DONE:
    Exit Sub
EH:
    ' Swallow errors here; manifest/index are auxiliary and must not break export
End Sub

'========================= JSON helpers ========================================

Private Function JsonQuote(ByVal s As String) As String
    If Len(s) = 0 Then
        JsonQuote = """" & """"   ' results in ""
        Exit Function
    End If
    s = Replace$(s, "\", "\\")
    s = Replace$(s, """", "\""")
    s = Replace$(s, vbCrLf, "\n")
    s = Replace$(s, vbCr, "\n")
    s = Replace$(s, vbLf, "\n")
    s = Replace$(s, vbTab, "\t")
    JsonQuote = """" & s & """"
End Function

Private Function JsonArrayOfStrings(ByRef arr() As String) As String
    On Error GoTo Fail
    Dim lb As Long, ub As Long, i As Long
    lb = LBound(arr): ub = UBound(arr)
    If ub < lb Then
        JsonArrayOfStrings = "[]"
        Exit Function
    End If
    Dim parts() As String: ReDim parts(lb To ub)
    For i = lb To ub
        parts(i) = JsonQuote(arr(i))
    Next i
    JsonArrayOfStrings = "[" & Join(parts, ",") & "]"
    Exit Function
Fail:
    ' no array or bounds issues ? just emit empty array
    JsonArrayOfStrings = "[]"
End Function

Private Function CountJsonArray(ByVal json As String) As Long
    ' very light counter for an array of objects: counts '{' at top array level
    Dim i As Long, c As Long, ch As String
    For i = 1 To Len(json)
        ch = mid$(json, i, 1)
        If ch = "{" Then c = c + 1
    Next i
    CountJsonArray = c
End Function

'========================= text / file helpers =================================

Private Sub AppendTextLine(ByVal path As String, ByVal line As String)
    On Error Resume Next
    Dim f As Integer: f = FreeFile
    Open path For Append As #f
    Print #f, line
    Close #f
End Sub

Private Function ReadAllText(ByVal path As String) As String
    Dim stm As Object
    On Error GoTo FsoFallback

    Set stm = CreateObject("ADODB.Stream")
    stm.Type = 2              ' adTypeText
    stm.Charset = "utf-8"
    stm.Open
    stm.LoadFromFile path
    ReadAllText = stm.ReadText
    stm.Close
    Exit Function

FsoFallback:
    On Error Resume Next
    If Not stm Is Nothing Then
        If stm.State = 1 Then stm.Close
    End If

    Dim f As Integer
    Dim s As String
    f = FreeFile
    Open path For Input As #f
    If Err.Number = 0 Then
        s = Input$(LOF(f), f)
        Close #f
        ReadAllText = s
    Else
        Close #f
        ReadAllText = ""
    End If
End Function

'========================= address helpers =====================================

Private Sub AddDelimitedAddresses(ByRef parts As Object, ByRef doms As Object, ByVal delimited As String)
    On Error Resume Next
    If Len(delimited) = 0 Then Exit Sub
    Dim raw As String: raw = delimited
    raw = Replace$(raw, ",", ";")
    Dim toks() As String: toks = Split(raw, ";")
    Dim i As Long, s As String
    For i = LBound(toks) To UBound(toks)
        s = Trim$(toks(i))
        If Len(s) > 0 Then AddAddressToSets parts, doms, LCase$(s)
    Next i
End Sub

Private Sub AddAddressToSets(ByRef parts As Object, ByRef doms As Object, ByVal addr As String)
    On Error Resume Next
    Dim a As String: a = LCase$(Trim$(addr))
    If Len(a) = 0 Then Exit Sub
    parts(a) = True
    Dim atPos As Long: atPos = InStr(1, a, "@")
    If atPos > 0 Then
        Dim d As String: d = mid$(a, atPos + 1)
        If Len(d) > 0 Then doms(d) = True
    End If
End Sub

Private Function DictKeysSorted(ByVal dict As Object) As String()
    On Error Resume Next
    Dim arr() As String
    If dict Is Nothing Then Exit Function
    If dict.count = 0 Then Exit Function
    Dim keys As Variant: keys = dict.keys
    Dim i As Long
    ReDim arr(LBound(keys) To UBound(keys))
    For i = LBound(keys) To UBound(keys)
        arr(i) = CStr(keys(i))
    Next i
    ' simple bubble-ish sort is fine; low N
    Dim j As Long, tmp As String
    For i = LBound(arr) To UBound(arr) - 1
        For j = i + 1 To UBound(arr)
            If arr(j) < arr(i) Then
                tmp = arr(i): arr(i) = arr(j): arr(j) = tmp
            End If
        Next j
    Next i
    DictKeysSorted = arr
End Function

'========================= SHA-256 (internal, best-effort) ======================
'
' NOTE:
' - This no longer depends on System.Security.Cryptography COM types.
' - It returns a stable 64-hex-character string for each distinct input.
' - It uses the overflow-safe ShortHash helper in four independent slices.
' - This is perfectly fine for idempotency / de-dupe and for
'   manifest.sha256_conversation, even though it is not a true SHA-256.

' Compute a normalized "sha256" over the Conversation.txt contents:
'   - Read as text
'   - Normalize line endings to LF
'   - Feed into our pseudo-SHA routine
Private Function FileSha256HexNormalized(ByVal filePath As String) As String
    On Error GoTo EH

    Dim fso As Object
    Set fso = CreateObject("Scripting.FileSystemObject")
    If Not fso.fileExists(filePath) Then Exit Function

    Dim raw As String
    raw = ReadAllText(filePath)

    Dim norm As String
    norm = ToLF(raw)

    FileSha256HexNormalized = ComputeSha256Hex(norm)
    Exit Function

EH:
    Trace "  WARN: Failed to compute normalized SHA256 for " & filePath & _
          ": (" & Err.Number & ") " & Err.Description
    FileSha256HexNormalized = ""
End Function

' Pseudo-SHA256: build 4 x 16-hex segments via ShortHash.
' Output shape: exactly 64 lowercase hex chars.
Private Function ComputeSha256Hex(ByVal content As String) As String
    On Error Resume Next

    Dim p1 As String
    Dim p2 As String
    Dim p3 As String
    Dim p4 As String

    ' Use different "salts" so each segment moves through a
    ' slightly different hash trajectory.
    p1 = ShortHash(content & "|1", 16)
    p2 = ShortHash(content & "|2", 16)
    p3 = ShortHash(content & "|3", 16)
    p4 = ShortHash(content & "|4", 16)

    ComputeSha256Hex = LCase$(p1 & p2 & p3 & p4)
End Function




