#include "windows_security_attributes.hpp"

WindowsSecurityAttributes::WindowsSecurityAttributes() {
    win_psecurity_descriptor_ = (PSECURITY_DESCRIPTOR) calloc(1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void**));

    //NOTE: PSID = pointer
    PSID* security_id_double_ptr = (PSID*) ((PBYTE) win_psecurity_descriptor_ + SECURITY_DESCRIPTOR_MIN_LENGTH);

    PACL* ppACL = (PACL*) ((PBYTE) security_id_double_ptr + sizeof(PSID*));

    InitializeSecurityDescriptor(win_psecurity_descriptor_, SECURITY_DESCRIPTOR_REVISION);

    SID_IDENTIFIER_AUTHORITY sid_identifier_authority = SECURITY_WORLD_SID_AUTHORITY;
    AllocateAndInitializeSid(&sid_identifier_authority, 1, SECURITY_WORLD_RID, 0, 0, 0, 0, 0, 0, 0, security_id_double_ptr);

    EXPLICIT_ACCESS explicit_access = ExplicitAccessInfo(security_id_double_ptr);

    SetEntriesInAcl(1, &explicit_access, NULL, ppACL);

    SetSecurityDescriptorDacl(win_psecurity_descriptor_, TRUE, *ppACL, FALSE);

    win_security_attributes_.nLength = sizeof(win_security_attributes_);
    win_security_attributes_.lpSecurityDescriptor = win_psecurity_descriptor_;
    win_security_attributes_.bInheritHandle = TRUE;
}

EXPLICIT_ACCESS WindowsSecurityAttributes::ExplicitAccessInfo(PSID* security_id_double_ptr) {
    EXPLICIT_ACCESS explicit_access;

    ZeroMemory(&explicit_access, sizeof(EXPLICIT_ACCESS));

    explicit_access.grfAccessPermissions = STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
    explicit_access.grfAccessMode = SET_ACCESS;
    explicit_access.grfInheritance = INHERIT_ONLY;
    explicit_access.Trustee.TrusteeForm = TRUSTEE_IS_SID;
    explicit_access.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
    explicit_access.Trustee.ptstrName = (LPTSTR) *security_id_double_ptr;

    return explicit_access;
}

SECURITY_ATTRIBUTES* WindowsSecurityAttributes::operator&() {
    return &win_security_attributes_;
}

WindowsSecurityAttributes::~WindowsSecurityAttributes() {
    PSID* security_id_double_ptr =
        (PSID*)((PBYTE)win_psecurity_descriptor_ + SECURITY_DESCRIPTOR_MIN_LENGTH);
    PACL* ppACL = (PACL*)((PBYTE)security_id_double_ptr + sizeof(PSID*));

    if (*security_id_double_ptr) {
        FreeSid(*security_id_double_ptr);
    }
    if (*ppACL) {
        LocalFree(*ppACL);
    }
    free(win_psecurity_descriptor_);
}