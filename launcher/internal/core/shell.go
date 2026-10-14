package core

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Review items 15 (copy the server address) and 24 (open a folder, or a file selected, in
// Explorer). Both act outside the launcher; the tests replace the OS calls.

// CopyText puts text on the Windows clipboard (item 15: the screen passes the address it shows).
func (l *Launcher) CopyText(text string) error {
	if strings.TrimSpace(text) == "" {
		return fmt.Errorf("nothing to copy")
	}
	return l.os.clipboard(text)
}

func explorerExe() string {
	if root := os.Getenv("SystemRoot"); root != "" {
		return filepath.Join(root, "explorer.exe")
	}
	return "explorer.exe"
}

// explorerCmdLine is the literal command line: explorer.exe "<dir>", or
// explorer.exe /select,"<file>" (the form Explorer parses; the quotes keep spaces).
func explorerCmdLine(path string, selectFile bool) string {
	if selectFile {
		return `explorer.exe /select,"` + path + `"`
	}
	return `explorer.exe "` + path + `"`
}

// OpenFolder opens a folder in Explorer as an outside process (item 24).
func (l *Launcher) OpenFolder(dir string) error {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	if st, err := os.Stat(abs); err != nil || !st.IsDir() {
		return fmt.Errorf("not a folder: %s", abs)
	}
	return l.os.launch(explorerExe(), explorerCmdLine(abs, false))
}

// RevealFile opens Explorer with the file selected, as an outside process (item 24).
func (l *Launcher) RevealFile(path string) error {
	abs, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	if !fileExists(abs) {
		return fmt.Errorf("not a file: %s", abs)
	}
	return l.os.launch(explorerExe(), explorerCmdLine(abs, true))
}
