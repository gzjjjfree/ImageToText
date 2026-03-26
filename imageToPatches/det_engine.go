package imageToPatches

import (
	"fmt"
	"image"
	"math"
	"sort"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"
)

// TextLine 代表检测到的一个文字行
type TextLine struct {
	Box   image.Rectangle // 在原图中的矩形框
	Score float32         // 检测得分
	Image image.Image     // 裁剪出的单行图片 (已转正)
}

// DetEngine 检测引擎
type DetEngine struct {
	Session      *ort.AdvancedSession
	InputTensor  ort.Value
	OutputTensor ort.Value
	OutputData   []float32 // 预留给结果的切片
}

// NewDetEngine 创建并初始化检测引擎
var (
	detEngineOnce   sync.Once
	GlobalDetEngine *DetEngine
)

// 获取（或初始化）全局引擎
func GetGlobalDetEngine() (*DetEngine, error) {
	var err error
	detEngineOnce.Do(func() {
		GlobalDetEngine, err = NewDetEngine()
	})
	return GlobalDetEngine, err
}

func NewDetEngine() (*DetEngine, error) {
	inputName := "x"
	outputName := detOutputName

	if len(detModelBytes) == 0 {
		return nil, fmt.Errorf("检测模型数据为空或已被释放")
	}

	// 1. 预分配内存（这些 Tensor 将随 Session 一直存在）
	inputData := make([]float32, 1*3*640*640)
	inputShape := ort.NewShape(1, 3, 640, 640)
	inputTensor, _ := ort.NewTensor(inputShape, inputData)

	outputData := make([]float32, 1*1*640*640)
	outputShape := ort.NewShape(1, 1, 640, 640)
	outputTensor, _ := ort.NewTensor(outputShape, outputData)

	// 2. 创建并绑定！
	// 注意：这里直接用 detModelBytes 初始化，并绑定持久化的 Tensor
	session, err := ort.NewAdvancedSessionWithONNXData(detModelBytes,
		[]string{inputName}, []string{outputName},
		[]ort.Value{inputTensor}, []ort.Value{outputTensor}, nil)

	if err != nil {
		return nil, err
	}

	// 关键：一旦 Session 创建成功，立即释放字节数组
	detModelBytes = nil

	return &DetEngine{
		Session:      session,
		InputTensor:  inputTensor,
		OutputTensor: outputTensor,
		OutputData:   outputData,
	}, nil
}
func (d *DetEngine) Destroy() {
	// 增加防御性判断
	if d == nil || d.Session == nil {
		return
	}
	d.Session.Destroy()
}

var (
	// 复用检测时的 visited 数组，减少内存分配
	visitedPool = sync.Pool{
		New: func() any {
			return make([]bool, detInputSize*detInputSize)
		},
	}

	// 用于缩放的原图缓存区
	resizeImgPool = sync.Pool{
		New: func() interface{} {
			return image.NewRGBA(image.Rect(0, 0, 640, 640))
		},
	}

	// 专门为 BFS 准备的零分配队列 (640*640个像素，每个存 X 和 Y 两个坐标)
	bfsQueuePool = sync.Pool{
		New: func() interface{} {
			return make([]int, 640*640*2)
		},
	}
)

// Detect 执行完整的检测流程
//func (d *DetEngine) Detect(srcImg image.Image) ([]TextLine, error) {
//	// 1. 预处理：将图片数据填入已经绑定好的 inputData
//	// 注意：这里需要直接修改 d.inputTensor 关联的底层数据
//	// 或者简单点，获取 Tensor 的 Slice 并填充
//	inputSlice := d.InputTensor.(*ort.Tensor[float32]).GetData()
//	ratioH, ratioW := fillPreprocessedData(srcImg, inputSlice)
//
//	// 2. 执行推理：此时 Run() 不带任何参数
//	err := d.Session.Run()
//	if err != nil {
//		return nil, fmt.Errorf("Det 推理失败: %v", err)
//	}
//
//	// 3. 直接使用 d.outputData 进行后处理，得到原图坐标的矩形框列表
//	boxes := dbPostProcess(d.OutputData, ratioH, ratioW, srcImg.Bounds())
//
//	// 4. 后续裁剪逻辑...
//	return boxesToLines(srcImg, boxes), nil
//}

// boxesToLines 根据矩形框列表从原图裁剪出对应的行图片，并返回 TextLine 列表
func boxesToLines(srcImg image.Image, boxes []image.Rectangle) []TextLine {
	var lines []TextLine

	// 用于 SubImage 的接口断言
	subImager, ok := srcImg.(interface {
		SubImage(r image.Rectangle) image.Image
	})
	if !ok {
		fmt.Println("警告：原图不支持 SubImage 操作")
		return nil
	}

	for _, box := range boxes {
		// 1. 确保矩形框在原图范围内 (取交集)
		intersected := box.Intersect(srcImg.Bounds())

		// 2. 过滤掉无效或太小的框
		if intersected.Dx() < 5 || intersected.Dy() < 5 {
			continue
		}

		// 3. 裁剪
		subImg := subImager.SubImage(intersected)

		lines = append(lines, TextLine{
			Box:   intersected,
			Image: subImg,
		})
	}

	// 可选：根据 Y 坐标对行进行排序，确保识别顺序是从上到下
	sort.Slice(lines, func(i, j int) bool {
		return lines[i].Box.Min.Y < lines[j].Box.Min.Y
	})

	return lines
}



// GetBoxes 只做推理并返回坐标框，不裁图
func (d *DetEngine) GetBoxes(srcImg image.Image) ([]image.Rectangle, error) {
	inputSlice := d.InputTensor.(*ort.Tensor[float32]).GetData()
	ratioH, ratioW := fillPreprocessedData(srcImg, inputSlice)

	err := d.Session.Run()
	if err != nil {
		return nil, fmt.Errorf("Det 推理失败: %v", err)
	}

	return dbPostProcess(d.OutputData, ratioH, ratioW, srcImg.Bounds()), nil
}

// dbPostProcess 使用优化后的 BFS
func dbPostProcess(probMap []float32, ratioH, ratioW float64, origBounds image.Rectangle) []image.Rectangle {
	const threshold = 0.3
	const minArea = 5

	visited := visitedPool.Get().([]bool)
	queue := bfsQueuePool.Get().([]int) // 获取预分配队列
	for i := range visited {
		visited[i] = false
	}
	defer visitedPool.Put(visited)
	defer bfsQueuePool.Put(queue) // 归还队列

	var boxes []image.Rectangle
	paddingH, paddingW := 6, 6

	for y := 0; y < 640; y++ {
		for x := 0; x < 640; x++ {
			if !visited[y*640+x] && probMap[y*640+x] > threshold {
				// 传入复用的 queue
				minX, minY, maxX, maxY, area := bfsOptimized(probMap, visited, x, y, threshold, queue)

				// 过滤掉过小的区域
				if area < minArea { continue }

				rect := image.Rect(
					int(math.Max(0, float64(minX)/ratioW-float64(paddingW))),
					int(math.Max(0, float64(minY)/ratioH-float64(paddingH))),
					int(math.Min(float64(origBounds.Dx()), float64(maxX)/ratioW+float64(paddingW))),
					int(math.Min(float64(origBounds.Dy()), float64(maxY)/ratioH+float64(paddingH))),
				)
				boxes = append(boxes, rect)
			}
		}
	}
	return boxes
}

// 零分配的高性能 BFS
func bfsOptimized(probMap []float32, visited []bool, startX, startY int, threshold float32, queue []int) (int, int, int, int, int) {
	minX, maxX := startX, startX
	minY, maxY := startY, startY
	area := 0

	// 模拟队列的头尾指针
	head, tail := 0, 0
	
	// 入队 (x, y)
	queue[tail] = startX; tail++
	queue[tail] = startY; tail++
	visited[startY*640+startX] = true

	// 提前定义方向，避免在循环内部申请内存
	dx := [4]int{0, 0, 1, -1}
	dy := [4]int{1, -1, 0, 0}

	for head < tail {
		// 出队
		x := queue[head]; head++
		y := queue[head]; head++
		area++

		// 更新边界
		if x < minX { minX = x }
		if x > maxX { maxX = x }
		if y < minY { minY = y }
		if y > maxY { maxY = y }

		// 检查 4 个邻居
		for i := 0; i < 4; i++ {
			nx, ny := x+dx[i], y+dy[i]
			// 边界与访问检查
			if nx >= 0 && nx < 640 && ny >= 0 && ny < 640 {
				idx := ny*640 + nx
				if !visited[idx] && probMap[idx] > threshold {
					visited[idx] = true
					// 入队
					queue[tail] = nx; tail++
					queue[tail] = ny; tail++
				}
			}
		}
	}
	return minX, minY, maxX, maxY, area
}

func (d *DetEngine) DetectAndFixOrientation(src image.Image) ([]TextLine, error) {
	// 1. 只获取框，不切图
	boxes, err := d.GetBoxes(src) 
	if err != nil { return nil, err }
	
	tallCount := 0
	for _, box := range boxes {
		if box.Dy() > box.Dx() {
			tallCount++
		}
	}

	// 2. 判断方向
	if len(boxes) > 0 && float32(tallCount)/float32(len(boxes)) > 0.5 {
		fmt.Println("检测到整图旋转，进行 90 度修正...")
		fixedImg := rotate90(src) 
		// 重新获取转正后的框
		boxes, err = d.GetBoxes(fixedImg) 
		if err != nil { return nil, err }
		// 3. 对修正后的图进行切分
		return boxesToLines(fixedImg, boxes), nil
	}
	
	// 3. 正常切分
	return boxesToLines(src, boxes), nil
}

// 修改 fillPreprocessedData 以复用图像内存
func fillPreprocessedData(srcImg image.Image, dest []float32) (ratioH, ratioW float64) {
	if srcImg == nil { return 0, 0 }
	bounds := srcImg.Bounds()
	
	ratioH = 640.0 / float64(bounds.Dy())
	ratioW = 640.0 / float64(bounds.Dx())

	// 复用内存
	resized := resizeImgPool.Get().(*image.RGBA)
	defer resizeImgPool.Put(resized)

	draw.BiLinear.Scale(resized, resized.Bounds(), srcImg, bounds, draw.Over, nil)

	pix := resized.Pix 
	size := 640 * 640
	gOff, bOff := size, size*2 

	for i := 0; i < size; i++ {
		p := i * 4
		// 位运算比乘法快一点点，且不需要强制类型转换开销
		dest[i] = float32(pix[p])/127.5 - 1.0
		dest[i+gOff] = float32(pix[p+1])/127.5 - 1.0
		dest[i+bOff] = float32(pix[p+2])/127.5 - 1.0
	}
	return ratioH, ratioW
}
