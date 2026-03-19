package main

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"strings"
	"time"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/widget"
	"github.com/gzjjjfree/ImageToText/imageToPatches"
	ort "github.com/yalue/onnxruntime_go"
)

// 假设 TextLine 已定义
type TextLine struct {
	Text string
}

var version = "1.0.0"

func main() {
	myApp := app.NewWithID("com.gzjjj.imagetotext")
	myWindow := myApp.NewWindow("imageToText - v" + version)

	// 1. 改为 Entry 控件，支持编辑和自动换行
	//resultEntry := widget.NewEntry()
	resultEntry := widget.NewLabelWithStyle("结果将在这里显示...", fyne.TextAlignLeading, fyne.TextStyle{Italic: true})
	resultEntry.Wrapping = fyne.TextWrapBreak
	//resultEntry.PlaceHolder = "结果将在这里显示..."
	// 并且放入 Scroll 容器防止长文本超出屏幕
	resultScroll := container.NewVScroll(container.NewPadded(resultEntry))

	// 2. 定义状态/耗时标签 (只读)
	statusLabel := widget.NewLabelWithStyle("等待操作", fyne.TextAlignLeading, fyne.TextStyle{Italic: true})

	// 3. 保存文件函数
	saveBtn := widget.NewButton("保存到本地", func() {
		if resultEntry.Text == "" {
			dialog.ShowInformation("提示", "内容为空，无需保存", myWindow)
			return
		}

		// 弹出保存对话框
		d := dialog.NewFileSave(func(writer fyne.URIWriteCloser, err error) {
			if err != nil || writer == nil {
				return
			}
			defer writer.Close()

			_, err = writer.Write([]byte(resultEntry.Text))
			if err != nil {
				dialog.ShowError(err, myWindow)
			} else {
				// 保存成功提示
				statusLabel.SetText("文件保存成功: " + writer.URI().Name())
			}
		}, myWindow)

		d.SetFileName("识别结果_" + time.Now().Format("20060102_150405") + ".txt")
		d.Show()
	})

	openBtn := widget.NewButton("选择图片", func() {})

	//var duration time.Duration
	timetext := "" //duration.Seconds()

	openBtn = widget.NewButton("选择图片", func() {
		dialog.NewFileOpen(func(reader fyne.URIReadCloser, err error) {
			if err != nil {
				fyne.Do(func() {
					dialog.ShowError(err, myWindow)
					resultEntry.SetText("文件错误: " + err.Error())
				})
				return
			}

			if reader == nil {
				// 用户点击了取消按钮，直接返回，不要继续执行
				return
			}

			// 1. 先显示加载状态
			fyne.Do(func() {
				resultEntry.SetText("正在识别中，请稍候...")
				openBtn.Disable() // 识别期间禁用按钮
			})

			// --- 开始计时 ---
			startTime := time.Now()
			//statusTime := time.Now()
			// 解码图片
			//img, _, err := image.Decode(reader)
			//img, _, err := imageToPatches.Decode(reader)
			//duration = time.Since(statusTime)

			//timetext += " 检测:" + fmt.Sprintf("%.3fs", duration.Seconds())
			// 2. 开启协程处理耗时任务
			go func() {
				defer reader.Close()
				// 解码图片
				img, _, err := image.Decode(reader)
				if err != nil {
					fyne.Do(func() {
						resultEntry.SetText("图片解码失败: " + err.Error())
					})
					return
				}
				// 1. 初始化 Builder
				var sb strings.Builder
				//statusduration := time.Since(statusTime)
				//timetext = " 解码:" + fmt.Sprintf("%.3fs", statusduration.Seconds()) + "\n"
				//sb.WriteString(timetext)
				//statusTime := time.Now()

				lines := imageToPatches.ImageToPatches(img)

				// --- 计算总耗时 ---
				//statusduration = time.Since(statusTime)
				//timetext += " 识别:" + fmt.Sprintf("%.3fs", statusduration.Seconds())

				

				// 2. 在循环中写入
				for _, line := range lines {
					sb.WriteString(line.Text)
					sb.WriteString("\n")
				}

				// 3. 一次性获取最终结果
				finalText := sb.String()

				duration := time.Since(startTime)
				//timetext += " 识别:" + fmt.Sprintf("%.3fs", duration.Seconds())

				fyne.Do(func() {
					// 耗时只显示在状态栏
					statusLabel.SetText(fmt.Sprintf("识别成功 | 耗时: %.3fs", duration.Seconds()))
					//statusLabel.SetText(timetext)
					// 结果框只存文字
					resultEntry.SetText(finalText)
					openBtn.Enable()
				})

				// 3. 回到主线程更新 UI
				fyne.Do(func() {
					myWindow.Canvas().Refresh(resultEntry)
					resultEntry.SetText(finalText)
					openBtn.Enable()
				})
			}()
		}, myWindow).Show()
	})

	openBtn.Enable()

	copyBtn := widget.NewButton("一键复制", func() {
		myApp.Clipboard().SetContent(resultEntry.Text)
		statusLabel.SetText("内容已复制到剪贴板")
	})

	// 顶部放操作按钮，底部放状态栏
	topBtns := container.NewHBox(openBtn, saveBtn, copyBtn)
	bottomBar := container.NewHBox(statusLabel)

	content := container.NewBorder(
		topBtns,      // 顶部按钮
		bottomBar,    // 底部
		nil,          // 左侧
		nil,          // 右侧
		resultScroll, // 主内容（文本框）
	)

	// --- 预热协程 ---
	go func() {
		statusTime := time.Now()
		// 延迟 1 秒，等 UI 完全渲染出来后再开始，避免抢占启动时的 CPU
		time.Sleep(time.Second * 1)

		fmt.Println("GO_LOG: 开始后台预热模型...")

		go imageToPatches.LoadDict(imageToPatches.DictPath)

		imageToPatches.InitORT()

		// 预分配一个足够大的缓冲区（对应宽度约 1000px）
		const prewarmSize = 128 * 18385
		imageToPatches.PrewarmLogitsPool(prewarmSize, 1)
		fmt.Println("GO_LOG: 内存池预热完成")

		engine, err := imageToPatches.GetGlobalDetEngine()
		if err != nil {
			fmt.Printf("GO_LOG: det预热失败: %v\n", err)
			return
		}
		if engine != nil && engine.Session != nil {
			err := engine.Session.Run() // 因为创建时绑定了 Tensor，这里直接 Run
			if err != nil {
				fmt.Printf("GO_LOG: Run det预热失败: %v\n", err)
				return
			}
		}

		_, err = imageToPatches.GetGlobalClsEngine()
		if err != nil {
			fmt.Printf("GO_LOG: cls预热失败: %v\n", err)
			return
		}

		// 3. 核心：预热识别模型 (Rec) —— 解决那 200ms 的关键！
		rec, _ := imageToPatches.GetGlobalRecEngine()
		if rec != nil {
			// 构造一个极小的空数据，触发一次推理
			fakeWidth := 80
			fakeData := make([]float32, 1*3*48*fakeWidth)
			// 调用一次 Recognize，让底层算子完成初始化
			_, err = rec.Recognize(fakeData, fakeWidth)
			if err != nil {
				fmt.Println("GO_LOG: 模型预热完成，现在识别将是满速状态")
				return
			}
		}

		duration := time.Since(statusTime)
		timetext += "预处理:" + fmt.Sprintf("%.3fs", duration.Seconds())
	}()

	defer func() {
		fmt.Println("GO_LOG: 正在关闭 App，清理资源...")

		// 1. 销毁检测引擎 (增加 nil 检查)
		// 建议直接在包里暴露一个专用的安全销毁函数，或者在这里判断
		if det, _ := imageToPatches.GetGlobalDetEngine(); det != nil {
			det.Destroy()
		}

		// 2. 销毁识别引擎 (如果你也做了单例)
		if rec, _ := imageToPatches.GetGlobalRecEngine(); rec != nil {
			// 记得给 RecEngine 也写一个 Destroy 方法
			rec.Session.Destroy()
		}

		//
		if cls, _ := imageToPatches.GetGlobalClsEngine(); cls != nil {
			// 记得给 RecEngine 也写一个 Destroy 方法
			cls.Session.Destroy()
		}

		// 3. 最后销毁环境
		ort.DestroyEnvironment()
	}()

	myWindow.SetContent(content)
	myWindow.Resize(fyne.NewSize(360, 640))
	myWindow.ShowAndRun()

}
