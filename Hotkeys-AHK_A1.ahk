; ====================================================
; AutoHotkey Script for Hotkey support in LongFormSTT
; ====================================================

#NoEnv
#SingleInstance Force
SendMode Input
SetWorkingDir %A_ScriptDir%

; ----------------------------------------------------------
; Suppress F2, ... at the Windows level
; (*) means capture with all modifiers, e.g. Shift+F2, etc.
; If you want them strictly with no modifiers, you could use:
;   F2::
;   ...
;
; But typically *F2:: is guaranteed to catch and suppress them.
; ----------------------------------------------------------


*F2::
    ; Send message "TOGGLE_LANGUAGE" to Python server running on localhost:34909
    ; Use the Hide parameter (3rd argument). No console window is shown.
    RunWait, %comspec% /c echo TOGGLE_LANGUAGE | ncat 127.0.0.1 34909,, Hide
return

*F3::
    RunWait, %comspec% /c echo START_RECORDING | ncat 127.0.0.1 34909,, Hide
return

*F4::
    RunWait, %comspec% /c echo STOP_AND_TRANSCRIBE | ncat 127.0.0.1 34909,, Hide
return

*F5::
    RunWait, %comspec% /c echo TOGGLE_ENTER | ncat 127.0.0.1 34909,, Hide
return

*F6::
    RunWait, %comspec% /c echo QUIT | ncat 127.0.0.1 34909,, Hide
return

*F10::
    RunWait, %comspec% /c echo TRANSCRIBE_STATIC | ncat 127.0.0.1 34909,, Hide
return
