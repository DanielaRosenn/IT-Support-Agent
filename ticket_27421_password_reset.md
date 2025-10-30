THIS IS A MEMORY EXAMPLE
# Ticket #27421: Can't Remember Password - Windows Login Not Working

## Ticket Information

**Ticket ID:** 27421
**Requester:** Ofir (ofir@catonetworks.com)
**Category:** Access and Security
**Subject:** Reset Windows password - forgot password link not working

**Description:**
```
The ticket requester: Ofir
-- Requester Email Address: ofir@catonetworks.com
The forgot my password for the Windows login isn't working
```

---

## Problem Summary

User cannot log into Windows laptop because they forgot their password. 
The "Forgot Password" option on the Windows login screen isn't working properly.

**Common symptoms:**
- Cannot log into Windows device due to forgotten password
- Forgot password link on login screen not responding
- Locked out of Windows after multiple failed attempts
- Need password reset but login screen options aren't working

---

## IT Response

Hi Ofir,

You can reset your password using Microsoft's Self-Service Password Reset (SSPR) portal.

### Steps to Reset Your Password:

1. **Visit the password reset portal:**
   https://passwordreset.microsoftonline.com/

2. **Enter your work email:**
   ofir@catonetworks.com

3. **Complete the CAPTCHA and click Next**

4. **Verify your identity** using one of these options:
   - Text message with verification code to your mobile phone
   - Authenticator app notification

5. **Choose your new password**

**Important for Windows laptops:** Your new password will sync automatically within 2-3 minutes. Wait a few minutes after resetting, then use your new password to log into Windows.

**Note:** If you're using a Mac laptop, this will only change your Microsoft 365 password (email, Teams, OneDrive). Your Mac login password stays the same. If you need to change your Mac login password as well, please let me know.

**For detailed step-by-step instructions with screenshots:**
https://catonetworks.freshservice.com/a/solutions/articles/24000065981

If you encounter any issues with the self-service reset, please reply and I'll help you directly.

Best regards,
**Cato Networks IT Service Desk**

---

## Solution Details

### What is SSPR?

Self-Service Password Reset (SSPR) is a Microsoft feature that lets you reset your password independently without IT support. You can regain access when locked out or if you forget your password by using secure verification methods like text message or authenticator app.

### Why the Windows "Forgot Password" link doesn't work:

The Windows login screen "Forgot Password" option is for local Windows accounts only. Since Cato Networks uses Microsoft 365 cloud authentication, you need to use the online portal instead: https://passwordreset.microsoftonline.com/

### Verification Methods:

**Text My Mobile Phone:**
- Enter your registered phone number
- Receive a text with verification code
- Enter code to proceed

**Authenticator App:**
- Click "Send Notification"
- Open Microsoft Authenticator app
- Approve by entering the number shown on screen

### After Password Reset:

**Windows laptops:** Password syncs automatically in 2-3 minutes. Your Windows login password becomes the same as your new Microsoft password.

**Mac laptops:** Only your Microsoft 365 password changes. Mac login password remains unchanged. Contact IT if you also need to change Mac login password.

---

## Troubleshooting

**Portal says email isn't recognized:**
- Verify you're using complete work email (not personal)
- Check for typos
- Contact IT if issue persists

**Don't have access to phone:**
- Contact IT for manual password reset
- IT can verify identity through alternative methods

**New password not working on Windows:**
- Wait 5 minutes for sync
- Check Caps Lock
- Restart laptop and try again
- Contact IT if still not working after 10 minutes

---

## Keywords

forgot password, windows login, password reset, cannot login, locked out, sspr, self service password reset, windows password, authentication, account locked, login issue, password not working, forgot my password, windows laptop

## Solution Type

**Self-Service** - User can complete independently without IT intervention

## Labels


`password-reset` `windows-login` `sspr` `self-service` `authentication` `login-issues` `account-access` `locked-out` `forgot-password` `troubleshooting` `how-to` `end-user`
