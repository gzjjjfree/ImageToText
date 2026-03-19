package imageToPatches

import (
	"fmt"
	"image"
	"runtime"
	"strings"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"
)

// RecognitionResult 代表每行文本的识别结果
type RecognitionResult struct {
	Index int
	Text  string
	Error error
}

type RecEngine struct {
	// 使用 DynamicAdvancedSession，它支持 RunWithBinding
	Session *ort.DynamicAdvancedSession
}

var (
	recEngineOnce   sync.Once
	globalRecEngine *RecEngine
)

// GetGlobalRecEngine 获取全局唯一的识别引擎
func GetGlobalRecEngine() (*RecEngine, error) {
	var err error
	recEngineOnce.Do(func() {
		globalRecEngine, err = NewRecEngine()
	})
	return globalRecEngine, err
}

// NewRecEngine 创建一个新的识别引擎实例
func NewRecEngine() (*RecEngine, error) {
	if len(recModelBytes) == 0 {
		return nil, fmt.Errorf("识别模型数据为空或已被释放")
	}

	// 2. 准备输出 Tensor (形状随宽度动态变化)
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("创建 SessionOptions 失败: %v", err)
	}
	defer options.Destroy()

	// 2. 核心性能调优：限制单任务线程数
	// 既然我们外面已经用了 Goroutine 并行，这里设为 1 可以大幅减少 CPU 上下文切换
	// Intra-Op	单个卷积层/矩阵运算等操作内部的线程数
	err = options.SetIntraOpNumThreads(1)
	if err != nil {
		fmt.Printf("Warning: SetIntraOpNumThreads failed: %v\n", err)
	}

	// 这里设置为 1 可以减少不同操作之间的上下文切换，因为我们每次只运行一个模型实例
	// Inter-Op	计算图中的分支结构（如并行的卷积层）之间的线程数
	err = options.SetInterOpNumThreads(1)
	if err != nil {
		fmt.Printf("Warning: SetInterOpNumThreads failed: %v\n", err)
	}

	// 1. 创建动态 Session
	// 输入和输出名称可以在此时设为 nil，因为 Binding 会在运行时处理它们
	s, err := ort.NewDynamicAdvancedSessionWithONNXData(recModelBytes, nil, nil, options)
	if err != nil {
		return nil, fmt.Errorf("创建动态 Rec Session 失败: %v", err)
	}

	// 2. 核心：一旦底层 C Session 创建成功，立即释放 Go 层的字节数组
	recModelBytes = nil

	return &RecEngine{Session: s}, nil
}

// 识别结果的缓存池，避免频繁申请巨大的 float32 切片
// 由于 OCR 宽度动态变化，我们池化一个“足够大”的缓冲区
var logitsPool = sync.Pool{
	New: func() any {
		// 预分配一个支持最大宽度（假设 1024px）的缓冲区
		// 1024/8 * 18385 = 4,706,560 个 float32
		return make([]float32, defaultBufferSize)
	},
}

// PrewarmLogitsPool 暴露给外部的预热接口
// size: 预期的缓冲区大小（float32 数量）
// count: 预先填充多少个对象到池中
func PrewarmLogitsPool(size int, count int) {
	for i := 0; i < count; i++ {
		logitsPool.Put(make([]float32, size))
	}
}

// 内部使用的获取方法（非导出）
func getLogitsBuffer(requiredSize int) []float32 {
	buf := logitsPool.Get().([]float32)
	if len(buf) < requiredSize {
		// 如果发现池子里的货太小，直接原地扩容（这种情况极少见）
		return make([]float32, requiredSize)
	}
	return buf[:requiredSize]
}

// 内部使用的归还方法（非导出）
func putLogitsBuffer(buf []float32) {
	logitsPool.Put(buf)
}

// Recognize 单行识别函数 (支持并行调用)
func (r *RecEngine) Recognize(inputData []float32, width int) (string, error) {
	// 1. 为本次识别创建独立的 Binding 对象
	binding, err := r.Session.CreateIoBinding()
	if err != nil {
		return "", err
	}
	defer binding.Destroy()

	// 2. 准备输入 Tensor
	// 输入 Tensor 的形状是 [1, 3, 48, width]，其中 width 是根据实际输入图像动态计算的
	inputShape := ort.NewShape(1, 3, recHeight, int64(width))
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return "", err
	}
	defer inputTensor.Destroy()

	// 绑定输入节点，PaddleOCR 的识别输入通常名为 "x"
	err = binding.BindInput("x", inputTensor)
	if err != nil {
		fmt.Printf("[DEBUG] 绑定输入失败: %v\n", err) // 添加此行
		return "", err
	}

	outputN := width / v5DownsampleRatio // PP-OCRv5 下采样倍数是 8
	vocabSize := len(charDict)           // 字典大小
	totalSize := outputN * vocabSize

	// 从池中拿内存，而不是 make
	logitsData := getLogitsBuffer(totalSize)
	// 确保函数结束（无论成功失败）都把这块珍贵的内存还回去
	defer putLogitsBuffer(logitsData) // 归还到池中
	// 将这个切片包装成 Tensor 并绑定
	outputShape := ort.NewShape(1, int64(outputN), int64(vocabSize))
	logitsTensor, err := ort.NewTensor(outputShape, logitsData)
	if err != nil {
		return "", err
	}
	defer logitsTensor.Destroy()

	// 绑定输出
	if err := binding.BindOutput(recOutputName, logitsTensor); err != nil {
		return "", err
	}

	// 4. 推理
	if err := r.Session.RunWithBinding(binding); err != nil {
		return "", err
	}

	// 5. 【核心优化】动态获取推理后的真实形状
	// GetBoundOutputValues 会返回推理后更新了 Shape 信息的 Tensor
	outputValues, err := binding.GetBoundOutputValues()
	if err != nil {
		return "", err
	}

	resultTensor := outputValues[0].(*ort.Tensor[float32])
	actualShape := resultTensor.GetShape() // 这就是你在 Netron 看到的 Reshape_524_o0__d2

	// 必须手动 Destroy 这些由 GetBoundOutputValues 生成的包装对象
	defer func() {
		for _, v := range outputValues {
			v.Destroy()
		}
	}()

	actualN := int(actualShape[1])

	// 6. 解码：只取模型真正输出的那部分数据
	// 哪怕 logitsData 后面有预留的空白，也绝对不会干扰结果
	return ctcDecode(logitsData[:actualN*vocabSize]), nil
}

// processParallel 并行处理多个文本行的识别
func processParallel(textLines []TextLine) []RecognitionResult {
	// 1. 获取全局识别引擎实例
	engine, err := GetGlobalRecEngine()
	if err != nil {
		return nil
	}
	clsEngine, err := GetGlobalClsEngine() // 获取分类引擎
	if err != nil {
		return nil
	}

	results := make([]RecognitionResult, len(textLines))

	// 1. 全局偏置标记
    var globalFlip int32 = 0 // 0: 未知, 1: 确定转正, 2: 确定翻转 180
    
    // 2. 抽样预判（前 10 行）
    sampleSize := 10
    if len(textLines) < sampleSize { sampleSize = len(textLines) }
    
    flipVotes := 0
    for i := 0; i < sampleSize; i++ {
        if clsEngine.ShouldRotate180(textLines[i].Image) {
            flipVotes++
        }
    }
    
    // 如果抽样中超过 80% 一致，则锁定全局状态
    if float32(flipVotes)/float32(sampleSize) > 0.8 {
        globalFlip = 2 // 全局翻转
    } else if float32(flipVotes)/float32(sampleSize) < 0.2 {
        globalFlip = 1 // 全局不翻转
    }

	var wg sync.WaitGroup

	// 限制并发协程数
	sem := make(chan struct{}, runtime.NumCPU())

	for i := range textLines {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			sem <- struct{}{}        // 获取令牌
			defer func() { <-sem }() // 释放令牌

			//statusTime := time.Now()

			img := textLines[idx].Image

			// 使用投票结果，避免重复推理
            if globalFlip == 2 {
                img = rotate180(img)
            } else if globalFlip == 0 {
                // 只有在无法确定全局状态时，才进行单行推理（兜底）
                if clsEngine.ShouldRotate180(img) {
                    img = rotate180(img)
                }
            }

			//if clsEngine.ShouldRotate180(img) {
			//	img = rotate180(img) // 如果倒了，转正它
			//}

			// 将子图转为 float32 [1, 3, 48, W]
			inputData, actualWidth := preprocessSingleLine(img)

			// 调用识别
			text, err := engine.Recognize(inputData, actualWidth)
			if err != nil {
				results[idx] = RecognitionResult{Text: "识别失败"}
				return
			}

			//statusduration := time.Since(statusTime)
			//timetext := " 识别耗时:" + fmt.Sprintf("%.3fs", statusduration.Seconds())

			results[idx] = RecognitionResult{Text: text}
		}(i)
	}

	wg.Wait()
	return results
}

// 专门为单行识别设计的预处理
func preprocessSingleLine(src image.Image) ([]float32, int) {
	const targetH = 48
	const maxW = 1280

	// 1. 计算缩放后的宽度，保持纵横比
	bounds := src.Bounds()
	ratio := float64(targetH) / float64(bounds.Dy())
	newW := int(float64(bounds.Dx()) * ratio)
	if newW > maxW {
		newW = maxW
	}

	// 2. 创建一个宽度对齐到 8 的 RGBA 图像，缩放并填充背景为白色
	alignedW := (newW + 7) &^ 7
	if alignedW < 32 {
		alignedW = 32
	}

	dst := image.NewRGBA(image.Rect(0, 0, alignedW, targetH))
	for i := range dst.Pix {
		dst.Pix[i] = 255
	}
	// 使用 BiLinear 进行缩放，比 NearestNeighbor 产生的边缘更平滑，有利于识别
	draw.BiLinear.Scale(dst, image.Rect(0, 0, newW, targetH), src, bounds, draw.Over, nil)

	// --- 适配 V5 的归一化处理 ---
	stride := dst.Stride
	pix := dst.Pix
	data := make([]float32, 3*targetH*alignedW)

	for y := 0; y < targetH; y++ {
		for x := 0; x < alignedW; x++ {
			baseIdx := y*stride + x*4

			// 直接获取 R, G, B
			r := float32(pix[baseIdx])
			g := float32(pix[baseIdx+1])
			b := float32(pix[baseIdx+2])

			// 方案 A：转灰度（最稳，适合大多数场景）
			gray := 0.299*r + 0.587*g + 0.114*b

			// V5 标准归一化公式：(val / 255.0 - 0.5) / 0.5  => val / 127.5 - 1.0
			val := gray/127.5 - 1.0

			// 写入 3 个通道 (CHW 格式)
			// 第一通道 (R)
			data[y*alignedW+x] = val
			// 第二通道 (G)
			data[y*alignedW+x+targetH*alignedW] = val
			// 第三通道 (B)
			data[y*alignedW+x+2*targetH*alignedW] = val
		}
	}

	return data, alignedW
}

// ctcDecode 实现 CTC 解码逻辑，输入是 [N, vocabSize] 的概率矩阵
func ctcDecode(logits []float32) string {
	vocabSize := len(charDict)
	if vocabSize == 0 {
		return ""
	}

	nSteps := len(logits) / vocabSize
	if nSteps == 0 {
		return ""
	}

	var sb strings.Builder
	// 适当调高预估字数，减少扩容次数
	sb.Grow(64)

	lastID := -1
	lastStep := -1

	for i := 0; i < nSteps; i++ {
		offset := i * vocabSize
		stepData := logits[offset : offset+vocabSize]

		// 1. ArgMax 寻找逻辑优化
		maxID := 0
		maxVal := stepData[0]

		// 循环展开加速
		j := 1
		for ; j <= vocabSize-4; j += 4 {
			// 一次加载 4 个值，减少循环开销
			v0, v1, v2, v3 := stepData[j], stepData[j+1], stepData[j+2], stepData[j+3]
			if v0 > maxVal {
				maxVal = v0
				maxID = j
			}
			if v1 > maxVal {
				maxVal = v1
				maxID = j + 1
			}
			if v2 > maxVal {
				maxVal = v2
				maxID = j + 2
			}
			if v3 > maxVal {
				maxVal = v3
				maxID = j + 3
			}
		}
		for ; j < vocabSize; j++ {
			if stepData[j] > maxVal {
				maxVal = stepData[j]
				maxID = j
			}
		}

		// 2. CTC 解码核心逻辑
		if maxID > 0 { // 0 通常是 blank
			if maxID != lastID {
				// 空格启发式逻辑
				if lastStep != -1 && (i-lastStep) > 6 {
					sb.WriteByte(' ')
				}

				if maxID < len(charDict) {
					sb.WriteString(charDict[maxID])
					lastStep = i
				}
			}
		}
		lastID = maxID
	}

	return sb.String()
}
