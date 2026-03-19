package imageToPatches

import (
	"bufio"
	"fmt"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)



// initORT 初始化 ONNX Runtime 环境
func InitORT() {
	libPath := "./lib/x86_64/x86/libonnxruntime.so" //wsl
	if runtime.GOOS == "android" {
		libPath = "libonnxruntime.so" // 直接写文件名，不要路径！

	}
	ort.SetSharedLibraryPath(libPath)
	err := ort.InitializeEnvironment()
	if err != nil {
		panic(err)
	}
}

var (
	charDict []string
	dictOnce sync.Once
)

// LoadDict 加载字符字典，使用 sync.Once 确保只加载一次
func LoadDict(path string) {
	dictOnce.Do(func() {
		file, err := resFS.Open(path)
		if err != nil {
			fmt.Printf("GO_LOG: 字典加载失败: %v\n", err)
			return
		}
		defer file.Close()

		// 预分配容量，减少 append 导致的内存重分配（Paddle 字典通常 6000-15000 行）
		charDict = make([]string, 0, 18385)
		charDict = append(charDict, "blank")

		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			charDict = append(charDict, scanner.Text())
		}

		charDict = append(charDict, " ")
		fmt.Printf("字典加载完成，实际字符数: %d\n", len(charDict))
	})
}
