#pragma once

#include <windows.h>
#include <AclAPI.h>

class WindowsSecurityAttributes {
public:
    WindowsSecurityAttributes() {
        win_psecurity_descriptor_ = (PSECURITY_DESCRIPTOR)calloc(1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void**));

        PSID* ppSID = (PSID*)((PBYTE)win_psecurity_descriptor_ + SECURITY_DESCRIPTOR_MIN_LENGTH);

        PACL* ppACL = (PACL*)((PBYTE)ppSID + sizeof(PSID*));

        InitializeSecurityDescriptor(win_psecurity_descriptor_, SECURITY_DESCRIPTOR_REVISION);

        SID_IDENTIFIER_AUTHORITY sid_identifier_authority = SECURITY_WORLD_SID_AUTHORITY;
        AllocateAndInitializeSid(&sid_identifier_authority, 1, SECURITY_WORLD_RID, 0, 0, 0, 0, 0, 0, 0, ppSID);

        EXPLICIT_ACCESS explicit_access;
        ZeroMemory(&explicit_access, sizeof(EXPLICIT_ACCESS));
        explicit_access.grfAccessPermissions = STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
        explicit_access.grfAccessMode = SET_ACCESS;
        explicit_access.grfInheritance = INHERIT_ONLY;
        explicit_access.Trustee.TrusteeForm = TRUSTEE_IS_SID;
        explicit_access.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
        explicit_access.Trustee.ptstrName = (LPTSTR)*ppSID;

        SetEntriesInAcl(1, &explicit_access, NULL, ppACL);

        SetSecurityDescriptorDacl(win_psecurity_descriptor_, TRUE, *ppACL, FALSE);

        win_security_attributes_.nLength = sizeof(win_security_attributes_);
        win_security_attributes_.lpSecurityDescriptor = win_psecurity_descriptor_;
        win_security_attributes_.bInheritHandle = TRUE;
    }

    SECURITY_ATTRIBUTES* operator&() {
        return &win_security_attributes_;
    }

    ~WindowsSecurityAttributes() {
        PSID* ppSID =
            (PSID*)((PBYTE)win_psecurity_descriptor_ + SECURITY_DESCRIPTOR_MIN_LENGTH);
        PACL* ppACL = (PACL*)((PBYTE)ppSID + sizeof(PSID*));

        if (*ppSID) {
            FreeSid(*ppSID);
        }
        if (*ppACL) {
            LocalFree(*ppACL);
        }
        free(win_psecurity_descriptor_);
    }
protected:
    SECURITY_ATTRIBUTES win_security_attributes_;
    PSECURITY_DESCRIPTOR win_psecurity_descriptor_;
};