# NFCRM-1:2025 Inventories (§6.3, §6.4, §6.5)

_Source: `D:\01_Projects\INTERSYMBOLIC-GRC\results\multisource_arg.json`_  
_Clause references: NFCRM-1:2025 §6.3 (assets), §6.4 (vulnerabilities), §6.5 (threats)_

## §6.3 Asset Inventory (33 entries)

| Asset ID | Asset type | NFCRM classification | Software linked |
|---|---|---|---|
| `SRV-001` | server | Sensitive system | vsftpd |
| `SRV-002` | server | Sensitive system | OpenSSH |
| `SRV-003` | server | Sensitive system | Linux telnetd |
| `SRV-004` | server | Sensitive system | Postfix SMTP |
| `SRV-005` | server | Sensitive system | ISC BIND |
| `SRV-006` | server | Sensitive system | Apache httpd |
| `SRV-007` | server | Sensitive system | PHP |
| `SRV-008` | server | Sensitive system | DVWA (Damn Vulnerable Web Application) |
| `SRV-009` | server | Sensitive system | OWASP Mutillidae II |
| `SRV-010` | server | Sensitive system | phpMyAdmin |
| `SRV-011` | server | Sensitive system | TWiki |
| `SRV-012` | server | Sensitive system | TikiWiki CMS |
| `SRV-013` | server | Sensitive system | rpcbind |
| `SRV-014` | server | Sensitive system | Samba |
| `SRV-015` | server | Sensitive system | netkit r-services (rexec/rlogin/rsh) |
| `SRV-016` | server | Sensitive system | Java RMI Registry |
| `SRV-017` | server | Sensitive system | ingreslock root backdoor |
| `SRV-018` | server | Sensitive system | NFS |
| `SRV-019` | server | Sensitive system | ProFTPD |
| `SRV-020` | server | Sensitive system | MySQL |
| `SRV-021` | server | Sensitive system | distccd |
| `SRV-022` | server | Sensitive system | PostgreSQL |
| `SRV-023` | server | Sensitive system | VNC |
| `SRV-024` | server | Sensitive system | X11 |
| `SRV-025` | server | Sensitive system | UnrealIRCd |
| `SRV-026` | server | Sensitive system | Apache Tomcat |
| `SRV-027` | server | Sensitive system | Ruby DRb (Distributed Ruby) |
| `SRV-028` | server | Sensitive system | Apache httpd |
| `SRV-029` | server | Sensitive system | OpenSSL |
| `SRV-030` | workstation | Endpoint workstation | Adobe Acrobat Reader |
| `SRV-031` | workstation | Endpoint workstation | — |
| `SRV-032` | server | Sensitive system | Windows SMBv1, IIS |
| `SRV-033` | server | Sensitive system | Windows SMBv1, IIS |

## §6.4 Vulnerability Inventory (28 entries)

| CVE ID | Severity | KEV? | Affects assets |
|---|---|---|---|
| CVE-2011-2523 | — | no | SW-vsftpd-2.3.4, SRV-001 |
| CVE-2008-0166 | — | no | SW-OpenSSH-4.7p1 Debian 8ubuntu1, SRV-002 |
| CVE-2001-0554 | — | no | SW-Linux_telnetd-netkit, SRV-003 |
| CVE-2009-0696 | — | no | SW-ISC_BIND-9.4.2, SRV-005 |
| CVE-2007-6750 | — | no | SW-Apache_httpd-2.2.8, SRV-006 |
| CVE-2011-3192 | — | no | SW-Apache_httpd-2.2.8, SRV-006 |
| CVE-2012-1823 | — | no | SW-PHP-5.2.4, SRV-007 |
| CVE-2012-2311 | — | no | SW-PHP-5.2.4, SRV-007 |
| CVE-2009-1151 | — | no | SW-phpMyAdmin-2.11.x, SRV-010 |
| CVE-2008-5305 | — | no | SW-TWiki-4.2.4, SRV-011 |
| CVE-2007-5423 | — | no | SW-TikiWiki_CMS-1.9.5, SRV-012 |
| CVE-2007-2447 | — | no | SW-Samba-3.0.20-Debian, SRV-014 |
| CVE-1999-0651 | — | no | SW-netkit_r-services_(rexec/rlogin/rsh)-0.17, SRV-015 |
| CVE-2011-3556 | — | no | SW-Java_RMI_Registry-GNU Classpath grmiregistry, SRV-016 |
| CVE-1999-0170 | — | no | SW-NFS-v2-v4, SRV-018 |
| CVE-2010-4221 | — | no | SW-ProFTPD-1.3.1, SRV-019 |
| CVE-2008-0226 | — | no | SW-MySQL-5.0.51a-3ubuntu5, SRV-020 |
| CVE-2004-2687 | — | no | SW-distccd-v1 (GCC 4.2.4), SRV-021 |
| CVE-2013-1904 | — | no | SW-PostgreSQL-8.3.0-8.3.7, SRV-022 |
| CVE-2006-2369 | — | no | SW-VNC-Protocol 3.3, SRV-023 |
| CVE-1999-0526 | — | no | SW-X11-Xorg 7.3, SRV-024 |
| CVE-2010-2075 | — | no | SW-UnrealIRCd-3.2.8.1, SRV-025 |
| CVE-2009-0580 | — | no | SW-Apache_Tomcat-5.5 (AJP/1.3 on port 8009), SRV-026 |
| CVE-2011-5330 | — | no | SW-Ruby_DRb_(Distributed_Ruby)-1.8, SRV-027 |
| CVE-2017-7679 | — | no | SW-Apache_httpd-2.4.18 (Ubuntu 16.04 default, 5s keep-alive), SRV-028 |
| CVE-2014-0160 | — | no | SW-OpenSSL-1.0.1f, SRV-029 |
| CVE-2010-0188 | — | no | SW-Adobe_Acrobat_Reader-9.0, SRV-030 |
| CVE-2017-0144 | — | no | SW-Windows_SMBv1-Windows Server 2012 R2, SRV-032, SW-Windows_SMBv1-Windows Server 2016 (+1) |

## §6.5 Threat Inventory (8 entries)

| Threat ID | Framework | Description | Targets |
|---|---|---|---|
| T1110 | MITRE ATT&CK | Brute Force | SRV-001, SRV-002 |
| T1110.001 | MITRE ATT&CK | Password Guessing (FTP/Patator) | SRV-001, SRV-002 |
| T1499 | MITRE ATT&CK | Endpoint Denial of Service | SRV-028 |
| T1498 | MITRE ATT&CK | Network Denial of Service | SRV-028 |
| T1555 | MITRE ATT&CK | Credentials from Password Stores | SRV-029 |
| T1190 | MITRE ATT&CK | Exploit Public-Facing Application | SRV-007, SRV-008 |
| T1566.001 | MITRE ATT&CK | Spearphishing Attachment (PDF) | SRV-030, SRV-031 |
| T1071.001 | MITRE ATT&CK | Web Protocols (Ares C2 port 8080) | SRV-032, SRV-033 |
