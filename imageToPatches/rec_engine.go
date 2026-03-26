package imageToPatches

import (
	"fmt"
	"image"
	"math"
	"runtime"
	"sort"
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

var (
	// 由于 OCR 宽度动态变化，我们池化一个“足够大”的缓冲区
	logitsPool = sync.Pool{
		New: func() any {
			// 预分配一个支持最大宽度（假设 1024px）的缓冲区
			// 1024/8 * 18385 = 4,706,560 个 float32
			return make([]float32, defaultBufferSize)
		},
	}
	// 用于 Rec 模型输入前，存放缩放后图像的池子
	recImagePool = sync.Pool{
		New: func() any {
			return image.NewRGBA(image.Rect(0, 0, recMaxW, recTargetH))
		},
	}

	// 用于 Rec 模型 ONNX 输入的 []float32 内存池
	// 尺寸: 3 通道 * 48 高度 * 最大宽度
	recFloatPool = sync.Pool{
		New: func() any {
			return make([]float32, 3*recTargetH*recMaxW)
		},
	}
)

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
	inputShape := ort.NewShape(1, 3, recTargetH, int64(width))
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

// preprocessRec 接收 SubImage 切图，并将其处理填入从池中借出的 destFloat 中
// 返回值 actualW 是缩放后的实际宽度，用于构建动态 Tensor
func preprocessRec(subImg image.Image, destFloat []float32) (actualW int) {
	bounds := subImg.Bounds()
	origW, origH := bounds.Dx(), bounds.Dy()

	// 1. 基于 48px 的目标高度计算宽度，保持原始纵横比
	tmpW := int(math.Round(float64(origW) * float64(recTargetH) / float64(origH)))

	// 计算总宽度：左边距 + 图像宽 + 右侧缓冲，并进行 8 像素对齐
	drawW := tmpW
	if tmpW > recMaxW {
		// 如果超过最大限制，强制压缩到 recMaxW，防止截断
		drawW = recMaxW
		actualW = recMaxW
	} else {
		// 没超过限制，按 8 像素对齐（PP-OCR 习惯）
		actualW = (tmpW + 7) &^ 7
		if actualW < 32 {
			actualW = 32
		}
	}

	// 2. 从池中借用一块大画布用于存放缩放结果
	tempImg := recImagePool.Get().(*image.RGBA)
	defer recImagePool.Put(tempImg)

	// 只使用大画布中实际需要的那部分区域
	fullRect := image.Rect(0, 0, actualW, recTargetH)
	draw.Draw(tempImg, fullRect, image.White, image.Point{}, draw.Src)

	// --- 3. 执行缩放 ---
	// targetRect 的宽度是 drawW。
	// 注意：这里需要先清空这一小块区域的背景（通常 OCR 模型习惯白底）
	// 因为 tempImg 是复用的，里面可能有上次的脏数据
	// 如果 drawW == recMaxW，说明发生了压缩，长句子会变瘦但会被完整保留。
	targetRect := image.Rect(0, 0, drawW, recTargetH)
	// 3. 执行缩放（注意：这里缩放到 tmpW，即保持原比例，右边留出对齐用的空白）
	//draw.BiLinear.Scale(tempImg, targetRect, subImg, bounds, draw.Over, nil)
	// 换用高质量缩放算法
	draw.CatmullRom.Scale(tempImg, targetRect, subImg, bounds, draw.Over, nil)
	// 4. 将像素转为 []float32 并填入 destFloat
	//pix := tempImg.Pix
	//stride := tempImg.Stride
	//channelSize := recTargetH * actualW
	//
	//for y := 0; y < recTargetH; y++ {
	//	rowOffset := y * stride
	//	for x := 0; x < actualW; x++ {
	//		p := rowOffset + x*4
	//
	//		// 归一化公式: (val / 127.5) - 1.0
	//		r := float32(pix[p])/127.5 - 1.0
	//		g := float32(pix[p+1])/127.5 - 1.0
	//		b := float32(pix[p+2])/127.5 - 1.0
	//		 //CHW 格式排列
	//		destFloat[y*actualW+x] = r
	//		destFloat[channelSize+y*actualW+x] = g
	//		destFloat[channelSize*2+y*actualW+x] = b
	//	}
	//}
	// 4. 将像素转为 []float32 并填入 destFloat
	pix := tempImg.Pix
	stride := tempImg.Stride
	channelSize := recTargetH * actualW
	c1 := channelSize
	c2 := channelSize * 2

	for y := 0; y < recTargetH; y++ {
		rowOffset := y * stride
		destRowOffset := y * actualW // 提前计算行偏移
		for x := 0; x < actualW; x++ {
			p := rowOffset + x<<2 // x*4 换成位移

			// 直接计算索引，减少乘法
			idx := destRowOffset + x

			// 归一化公式优化：提取常数
			const inv127 = 1.0 / 127.5
			destFloat[idx] = float32(pix[p])*inv127 - 1.0
			destFloat[idx+c1] = float32(pix[p+1])*inv127 - 1.0
			destFloat[idx+c2] = float32(pix[p+2])*inv127 - 1.0
		}
	}

	return actualW
}

func processParallel(textLines []TextLine) []RecognitionResult {
	//statusTime := time.Now()
	engine, err := GetGlobalRecEngine()
	if err != nil {
		return nil
	}
	// ... 获取 clsEngine 和前期投票判别逻辑不变 ...

	results := make([]RecognitionResult, len(textLines))

	clsEngine, err := GetGlobalClsEngine() // 获取分类引擎
	if err != nil {
		return nil
	}

	// 1. 全局偏置标记
	var globalFlip int32 = 0 // 0: 未知, 1: 确定转正, 2: 确定翻转 180

	// 2. 抽样预判（前 10 行）
	sampleSize := 10
	if len(textLines) < sampleSize {
		sampleSize = len(textLines)
	}

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

	// ==========================================
	// ⭐ 核心优化：基于方向和行容差进行智能排序
	// ==========================================

	isUpsideDown := (globalFlip == 2)

	// 2. 计算图像的水平中心点 (用于分栏)
	// 假设 textLines 已经包含了检测框信息，我们取所有框的中间位置作为参考
	imgCenterX := 0
	if len(textLines) > 0 {
		// 简单取所有框 X 范围的中点，或者直接传图片宽度进来
		minX, maxX := textLines[0].Box.Min.X, textLines[0].Box.Max.X
		for _, line := range textLines {
			if line.Box.Min.X < minX {
				minX = line.Box.Min.X
			}
			if line.Box.Max.X > maxX {
				maxX = line.Box.Max.X
			}
		}
		imgCenterX = (minX + maxX) / 2
	}

	// 3. 执行双栏智能排序
	sort.Slice(textLines, func(i, j int) bool {
		boxI := textLines[i].Box
		boxJ := textLines[j].Box

		centerI_X := boxI.Min.X + boxI.Dx()/2
		centerJ_X := boxJ.Min.X + boxJ.Dx()/2

		// --- 第一步：确定分栏逻辑 ---
		// 如果图片是倒转的(isUpsideDown)，左栏(Column 1)在视觉右侧，右栏(Column 2)在视觉左侧
		var colI, colJ int
		if isUpsideDown {
			if centerI_X > imgCenterX {
				colI = 1
			} else {
				colI = 2
			}
			if centerJ_X > imgCenterX {
				colJ = 1
			} else {
				colJ = 2
			}
		} else {
			if centerI_X < imgCenterX {
				colI = 1
			} else {
				colI = 2
			}
			if centerJ_X < imgCenterX {
				colJ = 1
			} else {
				colJ = 2
			}
		}

		// 如果不在同一栏，优先按栏排序
		if colI != colJ {
			return colI < colJ
		}

		// --- 第二步：在同一栏内进行行排序 (带容差) ---
		centerY_I := boxI.Min.Y + boxI.Dy()/2
		centerY_J := boxJ.Min.Y + boxJ.Dy()/2
		avgHeight := (boxI.Dy() + boxJ.Dy()) / 2

		// 同行判断
		if absInt(centerY_I-centerY_J) < avgHeight/2 {
			if isUpsideDown {
				return boxI.Min.X > boxJ.Min.X // 倒转时，行内从右往左
			}
			return boxI.Min.X < boxJ.Min.X // 正常时，行内从左往右
		}

		// 跨行判断
		if isUpsideDown {
			return centerY_I > centerY_J // 倒转时，从下往上读
		}
		return centerY_I < centerY_J // 正常时，从上往下读
	})
	// ==========================================

	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())

	for i := range textLines {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			img := textLines[idx].Image // 这就是那个几乎无开销的 SubImage

			// 根据方向投票结果决定是否旋转...
			// if globalFlip == 2 { img = rotate180(img) }
			// 使用投票结果，避免重复推理
			if globalFlip == 2 {
				img = rotate180(img)
			} else if globalFlip == 0 {
				// 只有在无法确定全局状态时，才进行单行推理（兜底）
				if clsEngine.ShouldRotate180(img) {
					img = rotate180(img)
				}
			}

			// 1. 核心：从池中获取复用的 float32 切片
			tensorData := recFloatPool.Get().([]float32)
			// 注意：必须在整个过程（包括推理）结束后再归还
			defer recFloatPool.Put(tensorData)

			// 2. 预处理，只使用 tensorData 的前 [3 * 48 * actualWidth] 部分
			actualWidth := preprocessRec(img, tensorData)

			// 3. 截取实际有效的数据长度传给 Rec 引擎
			// 因为 ONNX Runtime 需要严格匹配 Tensor 的尺寸
			validDataSize := 3 * recTargetH * actualWidth
			validData := tensorData[:validDataSize]

			// 4. 调用识别引擎 (你现有的 RecEngine 需要能接收 validData 和 actualWidth)
			text, err := engine.Recognize(validData, actualWidth)
			if err != nil {
				results[idx] = RecognitionResult{Text: "识别失败"}
				return
			}

			results[idx] = RecognitionResult{Text: text}
		}(i)
	}

	wg.Wait()
	//statusduration := time.Since(statusTime)
	//fmt.Println("识别: %.3fs", statusduration.Seconds())
	return results
}

// 辅助函数：求绝对值
func absInt(n int) int {
	if n < 0 {
		return -n
	}
	return n
}
