package imageToPatches

import (
	"fmt"
	"image"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"
)

type ClsEngine struct {
	Session *ort.DynamicAdvancedSession
}

var (
	clsEngineOnce   sync.Once
	globalClsEngine *ClsEngine
)

func GetGlobalClsEngine() (*ClsEngine, error) {
	var err error
	clsEngineOnce.Do(func() {
		globalClsEngine, err = NewClsEngine()
	})
	return globalClsEngine, err
}

var (
	// 管理预处理后的 float32 数据 [1, 3, 48, 192]
	clsInputPool = sync.Pool{
		New: func() interface{} {
			return make([]float32, 3*48*192)
		},
	}

	// 管理缩放用的临时图像对象
	clsImagePool = sync.Pool{
		New: func() interface{} {
			return image.NewRGBA(image.Rect(0, 0, 192, 48))
		},
	}
)

func NewClsEngine() (*ClsEngine, error) {
	if len(recModelBytes) == 0 {
		return nil, fmt.Errorf("识别模型数据为空或已被释放")
	}
	// 输入和输出名称可以在此时设为 nil，因为 Binding 会在运行时处理它们
	s, err := ort.NewDynamicAdvancedSessionWithONNXData(clsModelBytes, nil, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("创建动态 Rec Session 失败: %v", err)
	}

	// 2. 核心：一旦底层 C Session 创建成功，立即释放 Go 层的字节数组
	clsModelBytes = nil

	return &ClsEngine{Session: s}, nil
}

// Predict 判断是否需要翻转 (返回 true 表示需要旋转 180 度)
func (c *ClsEngine) ShouldRotate180(img image.Image) bool {
	// 1. 预处理到 [1, 3, 48, 192]
	//inputData := preprocessCls(img)

	// 1. 从池中获取 float32 缓冲区
	inputData := clsInputPool.Get().([]float32)
	defer clsInputPool.Put(inputData)
	preprocessCls(img, inputData)
	
	// 1. 为本次识别创建独立的 Binding 对象
	binding, err := c.Session.CreateIoBinding()
	if err != nil {
		return false
	}
	defer binding.Destroy()

	// 2. 准备输入 Tensor
	inputShape := ort.NewShape(1, 3, 48, 192)
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return false
	}
	defer inputTensor.Destroy()

	// 绑定输入节点，PaddleOCR 的识别输入通常名为 "x"
	err = binding.BindInput("x", inputTensor)
	if err != nil {
		fmt.Printf("[DEBUG] 绑定输入失败: %v\n", err) // 添加此行
		return false
	}

	// 3. 绑定输出节点（让 ONNX Runtime 自动分配输出内存）
	outputName := clsOutputName

	// 输出 [1, 2]，代表 0度 和 180度的概率
	outputShape := ort.NewShape(1, 2)
	outputData := make([]float32, 2)
	
	// 这里的 logitsData 是一个预分配的切片，ONNX Runtime 会直接写入这个切片的数据
	// 创建一个由现有数据切片支持的新张量。传递给此函数的形状会被复制，且在该函数返回后便不再需要。
	outputTensor, err := ort.NewTensor(outputShape, outputData)
	if err != nil {
		return false
	}
	defer outputTensor.Destroy()

	err = binding.BindOutput(outputName, outputTensor)
	if err != nil {
		fmt.Printf("[DEBUG] 绑定输出失败 (检查节点名): %v\n", err) // 添加此行
		return false
	}

	// 4. 执行推理 (RunWithBinding 是并发安全的)
	err = c.Session.RunWithBinding(binding)
	if err != nil {
		fmt.Printf("[DEBUG] cls推理执行失败: %v\n", err) // 添加此行
		return false
	}
	
	// 此时 outputData 已经被填充了，直接判断
	return outputData[1] > 0.5 && outputData[1] > outputData[0]
}

// 简单的 180 度旋转函数
func rotate180(src image.Image) image.Image {
	bounds := src.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	
	// 确保是 RGBA，如果不是则先转为 RGBA
	srcRGBA, ok := src.(*image.RGBA)
	if !ok {
		srcRGBA = image.NewRGBA(bounds)
		draw.Draw(srcRGBA, bounds, src, bounds.Min, draw.Src)
	}

	dst := image.NewRGBA(image.Rect(0, 0, w, h))

	// 直接通过 Pix 切片进行反向拷贝
	for y := 0; y < h; y++ {
		srcRow := srcRGBA.Pix[y*srcRGBA.Stride : y*srcRGBA.Stride+w*4]
		dstRow := dst.Pix[(h-y-1)*dst.Stride : (h-y-1)*dst.Stride+w*4]
		for x := 0; x < w; x++ {
			si := x * 4
			di := (w - x - 1) * 4
			// 4字节批量搬运 (RGBA)
			s := srcRow[si : si+4]
			d := dstRow[di : di+4]
			d[0], d[1], d[2], d[3] = s[0], s[1], s[2], s[3]
		}
	}
	return dst
}

// Rotate90 顺时针旋转 90 度
func rotate90(src image.Image) *image.RGBA {
	// 1. 确保输入是 *image.RGBA，如果不是则转换
	srcRGBA, ok := src.(*image.RGBA)
	if !ok {
		b := src.Bounds()
		srcRGBA = image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
		draw.Draw(srcRGBA, srcRGBA.Bounds(), src, b.Min, draw.Src)
	}

	srcBounds := srcRGBA.Bounds()
	sw, sh := srcBounds.Dx(), srcBounds.Dy()

	// 2. 创建目标图像：宽高互换
	dst := image.NewRGBA(image.Rect(0, 0, sh, sw))
	
	// 预提取切片，减少循环内的结构体访问
	srcPix := srcRGBA.Pix
	dstPix := dst.Pix
	srcStride := srcRGBA.Stride
	dstStride := dst.Stride

	// 3. 核心像素搬运逻辑
	// 外层遍历原图的 y，内层遍历 x
	for y := 0; y < sh; y++ {
		srcRowOffset := y * srcStride
		for x := 0; x < sw; x++ {
			// 原图坐标 (x, y) 的偏移量
			si := srcRowOffset + x*4
			
			// 旋转后坐标 (x', y') = (sh - y - 1, x)
			// 计算目标图的偏移量
			di := x*dstStride + (sh-y-1)*4
			
			// 直接搬运 4 字节 (R, G, B, A)
			// 使用这种写法可以触发编译器的优化，避免 slice 边界检查
			s := srcPix[si : si+4 : si+4]
			d := dstPix[di : di+4 : di+4]
			d[0] = s[0]
			d[1] = s[1]
			d[2] = s[2]
			d[3] = s[3]
		}
	}

	return dst
}

func preprocessCls(src image.Image, data []float32) []float32 {
	const targetH, targetW = 48, 192

	// 从池中获取临时图像
	dst := clsImagePool.Get().(*image.RGBA)
	defer clsImagePool.Put(dst)
	//dst := image.NewRGBA(image.Rect(0, 0, targetW, targetH))
	// 填充白色背景并缩放
	for i := range dst.Pix {
		dst.Pix[i] = 255
	}
	draw.BiLinear.Scale(dst, dst.Bounds(), src, src.Bounds(), draw.Over, nil)

	//data := make([]float32, 3*targetH*targetW)
	pix := dst.Pix
	for i := 0; i < targetH*targetW; i++ {
		p := i * 4
		// 归一化到 [-1, 1]
		val := (0.299*float32(pix[p])+0.587*float32(pix[p+1])+0.114*float32(pix[p+2]))/127.5 - 1.0
		data[i] = val
		data[i+targetH*targetW] = val
		data[i+2*targetH*targetW] = val
	}
	return data
}
