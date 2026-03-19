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

// 复用检测时的 visited 数组，减少内存分配
var visitedPool = sync.Pool{
	New: func() any {
		return make([]bool, detInputSize*detInputSize)
	},
}

// Detect 执行完整的检测流程
func (d *DetEngine) Detect(srcImg image.Image) ([]TextLine, error) {
	// 1. 预处理：将图片数据填入已经绑定好的 inputData
	// 注意：这里需要直接修改 d.inputTensor 关联的底层数据
	// 或者简单点，获取 Tensor 的 Slice 并填充
	inputSlice := d.InputTensor.(*ort.Tensor[float32]).GetData()
	ratioH, ratioW := fillPreprocessedData(srcImg, inputSlice)

	// 2. 执行推理：此时 Run() 不带任何参数
	err := d.Session.Run()
	if err != nil {
		return nil, fmt.Errorf("Det 推理失败: %v", err)
	}

	// 3. 直接使用 d.outputData 进行后处理，得到原图坐标的矩形框列表
	boxes := dbPostProcess(d.OutputData, ratioH, ratioW, srcImg.Bounds())

	// 4. 后续裁剪逻辑...
	return boxesToLines(srcImg, boxes), nil
}

// fillPreprocessedData 将原图预处理并填充到 dest 切片中，同时返回缩放比例
func fillPreprocessedData(srcImg image.Image, dest []float32) (ratioH, ratioW float64) {
	// 增加防御性检查
	if srcImg == nil {
		return 0, 0
	}
	// 1. 获取原图尺寸
	bounds := srcImg.Bounds()
	origH, origW := bounds.Dy(), bounds.Dx()

	// 2. 计算缩放比例
	ratioH = float64(detInputSize) / float64(origH)
	ratioW = float64(detInputSize) / float64(origW)

	// 3. 缩放图片到 640x640
	resized := image.NewRGBA(image.Rect(0, 0, detInputSize, detInputSize))
	// 注意：draw.BiLinear.Scale 的参数顺序是 (dst, dstRect, src, srcRect, op, opts)
	// 利用最近邻插值算法（BiLinear Interpolation）对图像进行缩放（Scaling）或拉伸（Resizing）。
	// 它通过对输入图像中的像素进行加权平均来计算输出图像中每个像素的值，从而实现平滑的缩放效果。
	draw.NearestNeighbor.Scale(resized, resized.Bounds(), srcImg, bounds, draw.Over, nil)

	pix := resized.Pix // 获取底层字节切片 [R, G, B, A, R, G, B, A...]
	size := 640 * 640
	gOff := size     // Green 存储偏移
	bOff := size * 2 // Blue 存储偏移

	for i := 0; i < size; i++ {
		// 每个像素占 4 字节 (RGBA)
		p := i * 4

		// (val / 255.0 - 0.5) / 0.5 简化后就是 (val / 127.5) - 1.0
		dest[i] = float32(pix[p])/127.5 - 1.0        // R
		dest[i+gOff] = float32(pix[p+1])/127.5 - 1.0 // G
		dest[i+bOff] = float32(pix[p+2])/127.5 - 1.0 // B
	}

	return ratioH, ratioW
}

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

// dbPostProcess 对概率图进行后处理，返回原图坐标的矩形框列表
func dbPostProcess(probMap []float32, ratioH, ratioW float64, origBounds image.Rectangle) []image.Rectangle {
	const threshold = 0.3
	const minArea = 16

	// 1. 创建二值图标记矩阵
	//visited := make([]bool, 640*640)
	visited := visitedPool.Get().([]bool)

	// 重置数组
	for i := range visited {
		visited[i] = false
	}
	defer visitedPool.Put(visited)

	var boxes []image.Rectangle

	// 在映射坐标时，给上下左右增加更合理的 padding
	paddingH := 8  // 纵向多给一点，防止切掉笔锋
	paddingW := 14 // 横向多给一点

	// 2. 遍历概率图，寻找未访问的“高概率”像素点
	for y := 0; y < 640; y++ {
		for x := 0; x < 640; x++ {
			if !visited[y*640+x] && probMap[y*640+x] > threshold {
				// 3. 发现新文字块，使用 BFS (广度优先搜索) 找完整轮廓
				minX, minY, maxX, maxY, area := bfs(probMap, visited, x, y, threshold)

				// 4. 过滤噪点
				if area < minArea {
					continue
				}

				// 5. 映射回原图坐标
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

// 简单的 BFS 寻找连通区域
func bfs(probMap []float32, visited []bool, startX, startY int, threshold float32) (int, int, int, int, int) {
	minX, maxX := startX, startX
	minY, maxY := startY, startY
	area := 0

	queue := [][2]int{{startX, startY}}
	visited[startY*640+startX] = true

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		x, y := curr[0], curr[1]
		area++

		if x < minX {
			minX = x
		}
		if x > maxX {
			maxX = x
		}
		if y < minY {
			minY = y
		}
		if y > maxY {
			maxY = y
		}

		// 检查上下左右 4 个邻居
		dirs := [][2]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
		for _, d := range dirs {
			nx, ny := x+d[0], y+d[1]
			if nx >= 0 && nx < 640 && ny >= 0 && ny < 640 && !visited[ny*640+nx] {
				if probMap[ny*640+nx] > threshold {
					visited[ny*640+nx] = true
					queue = append(queue, [2]int{nx, ny})
				}
			}
		}
	}
	return minX, minY, maxX, maxY, area
}

func (d *DetEngine) DetectAndFixOrientation(src image.Image) ([]TextLine, error) {
    lines, err := d.Detect(src) // 初次检测
	if err != nil {
		fmt.Printf("检测失败: %v\n", err)
		return nil, err
	}
    
    tallCount := 0
    for _, line := range lines {
        bounds := line.Image.Bounds()
        if bounds.Dy() > bounds.Dx() { // 高度大于宽度
            tallCount++
        }
    }

    // 如果超过 50% 的框是竖着的，说明整张大图需要预旋转 90 度
    if len(lines) > 0 && float32(tallCount)/float32(len(lines)) > 0.5 {
        fmt.Println("检测到整图旋转，正在进行 90 度预修正...")
        fixedImg := rotate90(src) // 实现一个 rotate90
        return d.Detect(fixedImg) // 重新检测转正后的图
    }
    
    return lines, nil
}
