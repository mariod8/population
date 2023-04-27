<template>
  <Map />
  <div class="absolute top-0 right-20 flex flex-col items-center justify-center h-screen">
    <input class="accent-grad1-1" type="range" min="1950" max="2021" v-model="year" />
    <Chart :data="data" :location="location" :year="year" />
  </div>
</template>
<script lang="ts">
import Map from "@/components/Map.vue"
import Chart from "@/components/Chart.vue"
import data from "@/data/data"
import type { Data, LocationData } from "@/types"

const defaultLocation = "Albania"

export default {
  data() {
    return {
      data: data as Data,
      location: defaultLocation as keyof Data,
      year: 2000 as keyof LocationData
    }
  },
  components: {
    Map,
    Chart
  }
}
</script>
