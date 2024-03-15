import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';
//import MutliBarGraphLegends from '../../../../Component/MutliBarGraphLegends';
//import * as d3 from "d3";
//import Chart from '../../../../Component/DrawBarChart';
import Chart from "react-apexcharts";
import Spinners from "../../../../spinneranimation";
import '../DemographyDashboard.scss';

const apiServices = new APIServices();

class DomIntRouteChart extends Component {
    constructor(props) {
        super();
        this.state = {
            top20RoutesData: [],
            colors: ["#0571b0", "#92c5de", '#46b6ed', "#4CAF50", "#d5d5d5", "#92c5de", "#0571b0"],
            headerName: 'TOP 20 DOMESTIC ROUTE SEARCH',
            isLoading: false,
            series: [],
            options: {
                legend: {
                    show: true
                },
                tooltip: {
                    style: {
                        fontSize: '10px',
                    },

                    fixed: {
                        enabled: true,
                        position: 'topRight',
                        offsetX: 0,
                        offsetY: 0,
                    }
                },
                chart: {
                    type: 'treemap',
                    offsetY: 0,
                    //height: 250,
                    toolbar: {
                        show: false
                    }
                },
                colors: [
                    '#3B93A5', '#F7B844', '#ADD8C7', '#EC3C65', '#CDD7B6', '#C1F666', '#D43F97', '#1E5D8C', '#421243', '#7F94B0', '#EF6537', '#C0ADDB'
                ],
                plotOptions: {
                    treemap: {
                        distributed: true,
                        enableShades: false
                    }
                }
            },
        };
    }

    componentWillReceiveProps = (props) => {
        this.getTileschartData(props);
    }

    getTileschartData = (props) => {
        const self = this;
        const { startDate, endDate, regionId, countryId, cityId } = props;
        if (self.state.headerName === 'TOP 20 DOMESTIC ROUTE SEARCH') {
            this.setState({ isLoading: true, top20RoutesData: [] })
            apiServices.getDemographyTop10Routes(startDate, endDate, regionId, countryId, cityId, 'Domestic').then((top20RoutesDataResponse) => {
                this.setState({ isLoading: false })
                if (top20RoutesDataResponse) {
                    const tableData = top20RoutesDataResponse[0].Data
                    if (tableData.length > 0) {
                        this.setState(
                            { top20RoutesData: this.normalizeData(tableData) }
                        );
                    }
                }
            });
        } else {
            this.setState({ isLoading: true })
            apiServices.getDemographyTop10Routes(startDate, endDate, regionId, countryId, cityId, 'International').then((top20RoutesDataResponse) => {
                this.setState({ isLoading: false, top20RoutesData: [] })
                if (top20RoutesDataResponse) {
                    const tableData = top20RoutesDataResponse[0].Data
                    if (tableData.length > 0) {
                        this.setState(
                            { top20RoutesData: this.normalizeData(tableData) }

                        );
                    }
                }
            });
        }
    }

    normalizeData = (data) => {
        let xValues = data.map(function (d) { return d.label; });
        let yValues = data.map(function (d) { return d.value; });
        let normalizedData = [];
        for (var i = 0; i < xValues.length; i++) {
            let categoryObj = { x: xValues[i].toUpperCase(), y: yValues[i] };
            normalizedData.push(categoryObj);
        }
        this.setState(
            {
                series: [
                    {
                        data: normalizedData
                    }
                ]
            }
        );
        return normalizedData;
    }

    headerChange(e) {
        this.setState({ headerName: e.target.value }, () => this.getTileschartData(this.props))
    }

    render() {
        const { isLoading, top20RoutesData } = this.state;
        console.log('dom states', this.state)
        return (
            <div className="x_panel tile" style={{ paddingTop: '0px', overflow: 'visible' }}>
                <div className="x_title reduce-margin">
                    <select class="select header-dropdown responsive-size" onChange={(e) => this.headerChange(e)}>
                        <option>TOP 20 DOMESTIC ROUTE SEARCH</option>
                        <option>TOP 20 INTERNATIONAL ROUTE SEARCH</option>
                    </select>
                    <ul className="nav navbar-right panel_toolbox">
                        <div className='info'><li><i class="fa fa-info" aria-hidden="true"></i></li>
                            {/*<MutliBarGraphLegends i={true} data={topfiveODsData} agentsName={agentsName} colors={colors} />*/}
                        </div>
                        <li onClick={() => this.props.history.push("/demographicReport")}><i className="fa fa-line-chart"></i></li>
                    </ul>
                </div>
                {isLoading ? <Spinners /> : top20RoutesData.length === 0 ?
                    <h5 style={{ textAlign: 'center', margin: '20%' }}>No data to show</h5> :
                    <div className='centered-graph'>
                        <div id="topODAgent"></div>
                        <Chart
                            options={this.state.options}
                            series={this.state.series}
                            type="treemap"
                            width="95%"
                            style={{ overflowY: "scroll" }}
                        />
                        {/*<MutliBarGraphLegends data={topfiveODsData} agentsName={agentsName} colors={colors} />*/}
                    </div>}
            </div>
        )
    }
}

export default DomIntRouteChart;