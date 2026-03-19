package imageToPatches

import (
	"embed"
	//_ "embed"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
)

//go:embed ppocrv5_dict.txt
var resFS embed.FS

//go:embed ch_ppocr_mobile_v2.0_cls_infer.onnx
var clsModelBytes []byte // 新增：分类模型

//go:embed ch_PP-OCRv5_mobile_det.onnx
var detModelBytes []byte

//go:embed ch_PP-OCRv5_rec_mobile_infer.onnx
var recModelBytes []byte

const (
	recOutputName = "fetch_name_0"                   // 识别模型的输出节点名称，已提取为全局变量
	detOutputName = "fetch_name_0"                   // 检测模型的输出节点名称，已提取为全局变量
	clsOutputName = "save_infer_model/scale_0.tmp_1" // 分类模型的输出节点名称，已提取为全局变量
	DictPath      = "ppocrv5_dict.txt"               // PP-OCRv5 的字典文件路径，已提取为全局变量
	//vocabSize = 18385	// PP-OCRv5 的字典大小，已提取为全局变量

	detInputSize = 640 // PP-OCRv3 默认是 640x640
	recHeight    = 48  // PP-OCRv5 识别输入高度固定为 48，宽度根据实际图像动态调整
	// V5 的下采样倍数。输入 48xW -> 输出 (W/4) 序列
	v5DownsampleRatio = 8

	// 默认初始化大小：256步 * 18385字典 ≈ 4.7M 个 float32 (约 18MB)
	defaultBufferSize = 128 * 18385
)

func ImageToPatches(srcImg image.Image) []RecognitionResult {
	if srcImg == nil {
		return nil
	}
	//startPrep := time.Now()

	// 初始化检测引擎 GetGlobalDetEngine
	detEngine, err := GetGlobalDetEngine()
	if err != nil {
		fmt.Printf("初始化检测引擎失败: %v\n", err)
		return nil
	}

	// 执行检测
	fmt.Println("正在检测文本行...")
	//textLines, err := detEngine.Detect(srcImg)
	textLines, err := detEngine.DetectAndFixOrientation(srcImg)
	if err != nil {
		fmt.Printf("检测失败: %v\n", err)
		return nil
	}

	//prepD := time.Since(startPrep)
	//timetext := "检测:" + fmt.Sprintf("%.3fs", prepD.Seconds()) + "\n"

	// 并行识别
	//results := processParallel(textLines)

	//results[0].Text = timetext + results[0].Text
	//prepD := time.Since(startPrep)
	//fmt.Printf("总耗时 %-10v\n", prepD.Round(time.Millisecond))

	//return results
	return processParallel(textLines)

}
