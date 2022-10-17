#pragma once

#include <AclAPI.h>

#include <stdexcept>

class WindowsSecurityAttributes {
public:
    WindowsSecurityAttributes();

    EXPLICIT_ACCESS ExplicitAccessInfo(PSID* security_id_double_ptr);

    ~WindowsSecurityAttributes();

    SECURITY_ATTRIBUTES* operator&();

    SECURITY_ATTRIBUTES SecurityAttributes() {
        return win_security_attributes_;
    }

protected:
    SECURITY_ATTRIBUTES win_security_attributes_ = {};
    PSECURITY_DESCRIPTOR win_psecurity_descriptor_ = NULL;
};