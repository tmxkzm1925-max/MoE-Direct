// Draft (GOL-DRAFT): editable, no completion stamp, never release use.

//go:build windows

package ui

import "golang.org/x/sys/windows"

// DisplayLanguage reads the Windows display language (the user's preferred
// UI languages) and returns LangKorean or LangEnglish.
func DisplayLanguage() string {
	names, err := windows.GetUserPreferredUILanguages(windows.MUI_LANGUAGE_NAME)
	if err != nil {
		return LangEnglish
	}
	return pickLang(names)
}
