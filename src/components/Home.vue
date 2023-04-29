<template>
  <Map @set-location="(loc: string) => location = loc" />
  <div
    class="absolute top-[50%] translate-y-[-50%] right-16 flex flex-col items-center justify-center w-[350px] bg-black-1 bg-opacity-75 p-5 rounded-3xl h-fit"
  >
    <input class="w-full opacity-80" type="range" min="1950" max="2021" v-model="year" />
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
<style>
/*********** Baseline, reset styles ***********/
input[type="range"] {
  -webkit-appearance: none;
  appearance: none;
  background: transparent;
  cursor: pointer;
}

/* Removes default focus */
input[type="range"]:focus {
  outline: none;
}

/******** Chrome, Safari, Opera and Edge Chromium styles ********/
/* slider track */
input[type="range"]::-webkit-slider-runnable-track {
  background-color: #eeeeee;
  border-radius: 0.5rem;
  height: 1.5rem;
}

/* slider thumb */
input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none; /* Override default look */
  appearance: none;
  margin-top: 4px; /* Centers thumb on the track */
  background-color: #393e46;
  border-radius: 0.5rem;
  height: 1rem;
  width: 1rem;
}

/*********** Firefox styles ***********/
/* slider track */
input[type="range"]::-moz-range-track {
  background-color: #eeeeee;
  border-radius: 0.5rem;
  height: 1.5rem;
  padding: 0 1rem;
}

/* slider thumb */
input[type="range"]::-moz-range-thumb {
  background-color: #393e46;
  border: none; /*Removes extra border that FF applies*/
  border-radius: 0.5rem;
  height: 1rem;
  width: 1rem;
}
</style>
