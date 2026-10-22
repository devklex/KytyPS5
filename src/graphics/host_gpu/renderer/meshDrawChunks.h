#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHDRAWCHUNKS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHDRAWCHUNKS_H_

#include <algorithm>
#include <cstdint>

namespace Libs::Graphics {

// Kyty's mesh shaders take their instance from the draw's push-data first instance plus
// WorkgroupId.y, so a mesh draw with more instances than one vkCmdDrawMeshTasksEXT accepts runs as
// several dispatches over instance ranges, each pushing its own first instance. Returns how many
// instances one dispatch of group_count_x groups may carry under the device's
// maxMeshWorkGroupCount[0..1] and maxMeshWorkGroupTotalCount, or 0 when not even one fits.
constexpr uint32_t MeshDrawInstancesPerDispatch(uint32_t group_count_x, uint32_t max_count_x,
                                                uint32_t max_count_y, uint32_t max_total_count) {
	if (group_count_x == 0 || group_count_x > max_count_x || group_count_x > max_total_count) {
		return 0;
	}
	return std::min(max_count_y, max_total_count / group_count_x);
}

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHDRAWCHUNKS_H_ */
