// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

//go:build !windows

package ui

import "os"

// DisplayLanguage reads LC_ALL, LC_MESSAGES or LANG outside Windows.
func DisplayLanguage() string {
	for _, k := range []string{"LC_ALL", "LC_MESSAGES", "LANG"} {
		if v := os.Getenv(k); v != "" {
			return pickLang([]string{v})
		}
	}
	return LangEnglish
}
