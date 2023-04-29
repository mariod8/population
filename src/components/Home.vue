<template>
  <Map @set-location="(loc: string) => location = loc" />
  <div
    class="absolute top-[50%] translate-y-[-50%] right-16 flex flex-col items-center justify-center w-[350px] bg-black-1 bg-opacity-75 p-5 rounded-3xl h-fit"
  >
    <input class="w-full" type="range" min="1950" max="2021" v-model="year" />
    <Chart :location="location" :data="data" :year="year" />
  </div>
</template>
<script lang="ts">
import Map from "@/components/Map.vue"
import Chart from "@/components/Chart.vue"
import data from "@/data/data"
import type { Data, LocationData } from "@/types"

export default {
  data() {
    return {
      data: data as Data,
      location: "W" as keyof Data,
      year: 2000 as keyof LocationData
    }
  },
  components: {
    Map,
    Chart
  }
}
</script>
