import React, { Component } from "react";
import APIServices from '../../../API/apiservices';
import Spinners from "../../../spinneranimation";


const apiServices = new APIServices();

class IndicatorsPRP extends Component {

    constructor(props) {
        super(props);
        this.state = {
            indicatorsDatas: [],
            indicatorsDivWidth: "",
            loading: false
        };
    }

    componentWillReceiveProps = (props) => {
        this.getIndicatorsData(props)
    }

    getIndicatorsData = (props) => {
        var self = this;
        const { startDate, endDate, routeGroup, regionId, countryId, cityId, routeId, dashboard } = props;
        self.setState({ loading: true, indicatorsDatas: [] })
        apiServices.getPRPIndicatorsData(startDate, endDate, routeGroup, regionId, countryId, cityId, routeId, dashboard).then((data) => {
            self.setState({ loading: false })
            if (data) {
                self.setState({
                    indicatorsDatas: data.slice(0, 3)
                })
            }
        });
    }

    arrowIndicator = (variance) => {
        // visually indicate if this months value is higher or lower than last months value
        let VLY = variance;
        if (VLY === '0') {
            return ''
        } else if (typeof VLY === 'string') {
            if (VLY.includes('B') || VLY.includes('M') || VLY.includes('K')) {
                return 'fa fa-arrow-up'
            } else {
                VLY = parseFloat(VLY)
                if (typeof VLY === 'number') {
                    if (VLY > 0) {
                        return 'fa fa-arrow-up'
                    } else {
                        return 'fa fa-arrow-down'
                    }
                }
            }
        } else {
            return ''
        }
    }

    renderIndicatorsFlow = () => {

    }


    render() {
        const { indicatorsDatas, loading } = this.state;
        return (
            loading ?
                <div className='no-indicators'><Spinners /></div> :
                indicatorsDatas.length === 0 ?
                    <h4 className='no-indicators'>No Indicators to show</h4> :
                    <div>
                        {indicatorsDatas.map((data, i) => {
                            return (
                                <div id="indicator_data" className="col-sm-4 col-xs-6 tile_stats_count" >
                                    <div className='indicator-top'>
                                        <div className="count_top"> {data.Name} </div>
                                        <div className="count" id={data.Name}> {window.numberFormat(data.CY)} </div>
                                    </div>
                                    <div className='indicator-bottom' >
                                        <span style={{ flex: '1' }}>{window.numberFormat(data.LY)} LY</span>
                                        <div className='indicator-2'>
                                            <i className={`${this.arrowIndicator(window.numberFormat(data.VLY))}`}></i>
                                            <span>{window.numberFormat(data.VLY)}% VLY</span>
                                        </div>
                                    </div>
                                    <div className='indicator-bottom' >
                                        <span style={{ flex: '1' }}>F</span>
                                        <div className='indicator-2'>
                                            <i className={`${this.arrowIndicator(window.numberFormat(data.VTG))}`}></i>
                                            <span >{window.numberFormat(data.VTG)}% VTG</span>
                                        </div>
                                    </div>
                                    <div className='indicator-bottom' >
                                        <span style={{ flex: '1' }}>J</span>
                                        <div className='indicator-2'>
                                            <i className={`${this.arrowIndicator(window.numberFormat(data.VTG))}`}></i>
                                            <span >{window.numberFormat(data.VTG)}% VTG</span>
                                        </div>
                                    </div>
                                    <div className='indicator-bottom' >
                                        <span style={{ flex: '1' }}>Y</span>
                                        <div className='indicator-2'>
                                            <i className={`${this.arrowIndicator(window.numberFormat(data.VTG))}`}></i>
                                            <span >{window.numberFormat(data.VTG)}% VTG</span>
                                        </div>
                                    </div>
                                </div>
                            )
                        })}
                        <div id="indicator_data" className="col-sm-4 col-xs-6 tile_stats_count" >
                            <div className='indicator-top'>
                                <div className="count_top"> {'Ex-Rate YTD'} </div>
                                <div className="count" id={'Ex-Rate YTD'}> {'2.3M'} </div>
                            </div>
                            <div className='indicator-bottom' >
                                <span style={{ flex: '1' }}>{'35'} LY</span>
                                <div className='indicator-2'>
                                    <i className={`${this.arrowIndicator(window.numberFormat('50'))}`}></i>
                                    <span>{'50'}% VLY</span>
                                </div>
                            </div>
                            <div className='indicator-bottom' >
                                <span style={{ flex: '1' }}>{'80'} BGT</span>
                                <div className='indicator-2'>
                                    <i className={`${this.arrowIndicator(window.numberFormat('40'))}`}></i>
                                    <span >{window.numberFormat('40')}% VTG</span>
                                </div>
                            </div>
                        </div>
                    </div>

        )

    }
}

export default IndicatorsPRP;
