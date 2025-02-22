; ====================================================
; AutoHotkey Script for Hotkey support in LongFormSTT
; ====================================================

#NoEnv
#SingleInstance Force
SendMode Input
SetWorkingDir %A_ScriptDir%

; ----------------------------------------------------------
; Suppress F2, F3, F4 at the Windows level
; (*) means capture with all modifiers, e.g. Shift+F2, etc.
; If you want them strictly with no modifiers, you could use:
;   F2::
;   F3::
;   F4::
; But typically *F2:: is guaranteed to catch and suppress them.
; ----------------------------------------------------------


*F2::
    ; Send message "F2" to Python server running on localhost:34909
    ; Use the Hide parameter (3rd argument). No console window is shown.
    RunWait, %comspec% /c echo F2 | ncat 127.0.0.1 34909,, Hide
return

*F3::
    RunWait, %comspec% /c echo F3 | ncat 127.0.0.1 34909,, Hide
return

*F4::
    RunWait, %comspec% /c echo F4 | ncat 127.0.0.1 34909,, Hide
return

*F5::
    RunWait, %comspec% /c echo QUIT | ncat 127.0.0.1 34909,, Hide
return

; ----------------------------------------------------------
; Optionally handle F2/F3/F4 *up* vs *down*, etc. 
; This is enough for simple press-and-release usage.
; ----------------------------------------------------------
